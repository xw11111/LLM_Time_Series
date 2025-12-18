import boto3
import io
import scipy
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tensorflow import keras
import neurokit2 as nk
import os
from src.Autoencoder import encode_waves
from collections import Counter
import warnings
import traceback
import importlib
from src import xresnet_embed
importlib.reload(xresnet_embed)
from src.xresnet_embed import WaveEmbeddingExtractor
warnings.simplefilter("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CONFIG = {
    "lead_index": 1,
    "dataset_name": "cpsc_2018",
    "sampling_rate": 500,
    "visualize": "no",
    "num_p_wave_clusters": 13,
    "num_qrs_wave_clusters": 11,
    "num_t_wave_clusters": 10,

}

suff = './pickles/new_pickles_800k_AE1/'

p_ae_path = suff +  "p_ae_model.pth"
qrs_ae_path = suff +  "qrs_ae_model.pth"
t_ae_path = suff +  "t_ae_model.pth"

p_cluster_path = suff +'p_wave_clustering_model.pkl'
qrs_cluster_path = suff +'qrs_wave_clustering_model.pkl'
t_cluster_path = suff +'t_wave_clustering_model.pkl'

dataset_name = CONFIG["dataset_name"]
finetune_path = suff + 'finetune/' + dataset_name
fine_tuning_sentences_file_name = finetune_path + '/beat_sentences_withID.pkl'
fine_tuning_sentences_cnn_file_name = finetune_path + '/beat_sentences_cnn_withID.pkl'
chunk_data_output_dir = finetune_path + "/patient_wave_data_chunks/"
patient_wave_data_file_name = finetune_path + "/patient_wave_data.pkl"
os.makedirs(finetune_path, exist_ok=True)

MODEL_PATH = "./models/fastai_xresnet1d101.pth"
extractor = WaveEmbeddingExtractor(
    model_path=MODEL_PATH,
    device=str(DEVICE),
    normalize=True
)

s3 = boto3.client('s3')
bucket_name = 'your_s3'
dataset_name = CONFIG["dataset_name"]
dataset_prefix = f"data/{dataset_name}/"

data_key = f"{dataset_prefix}REFERENCE.csv"

data_location = f"s3://{bucket_name}/{data_key}"

response = s3.get_object(Bucket=bucket_name, Key=data_key)
file_content = response['Body'].read()
df_main = pd.read_csv(io.BytesIO(file_content))

total_rows = len(df_main)
df_main = df_main[(df_main['Second_label'].isna() | (df_main['Second_label'] == '')) &
                  (df_main['Third_label'].isna()  | (df_main['Third_label']  == ''))].reset_index(drop=True)
dropped_rows = total_rows - len(df_main)
print(f"Dropped {dropped_rows} multi-label recordings; {len(df_main)} single-label recordings remain.")

df_main['filename'] = df_main['Recording'] + '.mat'

df_main['label'] = df_main['First_label']

leads = ['Lead I', 'Lead II', 'Lead III',
        'Lead aVR', 'Lead aVL', 'Lead aVF',
        'Lead V1', 'Lead V2', 'Lead V3',
        'Lead V4', 'Lead V5', 'Lead V6']

data_rrr_z = []
data_full_z = []

data_rrr= []
data_full = []

extra_lead_indices = np.arange(1,12)

folder = dataset_prefix

files_list = []
continuation_token = None

while True:

    if continuation_token:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder, ContinuationToken=continuation_token)
    else:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder)

    new_files = [obj['Key'].replace(folder, '') for obj in response.get('Contents', [])]
    files_list.extend(new_files)

    if response.get('IsTruncated'):
        continuation_token = response.get('NextContinuationToken')
    else:
        break

folder_names = [file.lstrip('/') for file in files_list if file.endswith(".mat")]

single_label_set = set(df_main['filename'])
before_filter = len(folder_names)
folder_names = [fn for fn in folder_names if os.path.basename(fn) in single_label_set]
after_filter = len(folder_names)

print(f"MAT files – before filter: {before_filter}; after single-label filter: {after_filter}")

sampling_rate = CONFIG["sampling_rate"]
visualize = CONFIG["visualize"]

def filter_indices(onset_offset_dict):
    arrays = np.array([
        onset_offset_dict['ECG_P_Onsets'],
        onset_offset_dict['ECG_P_Offsets'],
        onset_offset_dict['ECG_R_Onsets'],
        onset_offset_dict['ECG_R_Offsets'],
        onset_offset_dict['ECG_T_Onsets'],
        onset_offset_dict['ECG_T_Offsets']
    ])

    valid_indices = np.where(~np.isnan(arrays).any(axis=0))[0]

    valid_indices = [
        i for i in valid_indices
        if onset_offset_dict['ECG_T_Onsets'][i] <= onset_offset_dict['ECG_T_Offsets'][i]
    ]

    p_wave_indices = [
        [onset_offset_dict['ECG_P_Onsets'][i], onset_offset_dict['ECG_P_Offsets'][i]]
        for i in valid_indices
    ]
    qrs_indices = [
        [onset_offset_dict['ECG_R_Onsets'][i], onset_offset_dict['ECG_R_Offsets'][i]]
        for i in valid_indices
    ]
    t_wave_indices = [
        [onset_offset_dict['ECG_T_Onsets'][i], onset_offset_dict['ECG_T_Offsets'][i]]
        for i in valid_indices
    ]
    return p_wave_indices, qrs_indices, t_wave_indices

def align_delineation_with_rpeaks(rpeaks, delineation_points, category):
    aligned_delineation = [np.nan] * len(rpeaks)
    for delineation_point in delineation_points:
        if not np.isnan(delineation_point):
            distances = delineation_point - rpeaks
            if category == "precede":

                relevant_distances = np.where(distances < 0, np.abs(distances), np.inf)
            elif category == "follow":

                relevant_distances = np.where(distances > 0, distances, np.inf)
            else:
                raise ValueError("Invalid category. Use 'precede' or 'follow'.")
            closest_idx = np.argmin(relevant_distances)
            if relevant_distances[closest_idx] != np.inf: 
                aligned_delineation[closest_idx] = delineation_point
    return aligned_delineation

def prepare_wave_indices_for_plotting(waves_dwt):

    p_wave_indices = [
        (waves_dwt['ECG_P_Onsets'][i], waves_dwt['ECG_P_Offsets'][i])
        for i in range(len(waves_dwt['ECG_P_Onsets']))
    ]
    qrs_indices = [
        (waves_dwt['ECG_R_Onsets'][i], waves_dwt['ECG_R_Offsets'][i])
        for i in range(len(waves_dwt['ECG_R_Onsets']))
    ]
    t_wave_indices = [
        (waves_dwt['ECG_T_Onsets'][i], waves_dwt['ECG_T_Offsets'][i])
        for i in range(len(waves_dwt['ECG_T_Onsets']))
    ]

    return p_wave_indices, qrs_indices, t_wave_indices

def visualize_ecg_signal_with_rpeaks(ecg_signal, rpeaks):
    plt.figure(figsize=(10, 5))
    plt.plot(ecg_signal, label='ECG Signal')
    plt.plot(rpeaks, ecg_signal[rpeaks], 'ro', label='R-peaks')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with R-peaks')
    plt.legend()
    plt.show()

def visualize_ecg_with_waves(ecg_signal, p_wave_indices, qrs_indices, t_wave_indices):
    plt.figure(figsize=(10, 5))
    plt.plot(ecg_signal, label='ECG Signal')

    for start, end in p_wave_indices:
        plt.axvspan(start, end, color='red', alpha=0.3, label='P Wave' if start == p_wave_indices[0][0] else "")
    for start, end in qrs_indices:
        plt.axvspan(start, end, color='green', alpha=0.3, label='QRS Complex' if start == qrs_indices[0][0] else "")
    for start, end in t_wave_indices:
        plt.axvspan(start, end, color='blue', alpha=0.3, label='T Wave' if start == t_wave_indices[0][0] else "")

    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with P, QRS, and T Waves')
    plt.legend()
    plt.show()

def process_ecg_signal(ecg_signal, sampling_rate):

    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

    try:

        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

        rpeaks = rpeaks['ECG_R_Peaks']
        rpeaks = rpeaks[~np.isnan(rpeaks)].astype(int)

        if len(rpeaks) < 4:
            print("Not enough R-peaks detected to perform delineation.")
            return [], [], [], [], [], []

    except (ValueError, IndexError) as e:
        print(f"R-peak detection error: {e}")
        return [], [], [], [], [], []

    try:
        _, waves_dwt = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt")
    except ValueError as e:
        print(f"Delineation error: {e}")
        return [], [], [], [], [], []

    if visualize == 'yes':
        print('R peaks')
        visualize_ecg_signal_with_rpeaks(ecg_signal, rpeaks)

    waves_dwt_org = waves_dwt.copy()

    p_wave_indices_bad, qrs_indices_bad, t_wave_indices_bad = prepare_wave_indices_for_plotting(waves_dwt) 

    if visualize == 'yes':
        print('bad indices')
        visualize_ecg_with_waves(ecg_cleaned, p_wave_indices_bad, qrs_indices_bad,t_wave_indices_bad)
        print('p_wave_indices_bad',p_wave_indices_bad)

    category_mapping = {
    'ECG_P_Peaks': 'precede',
    'ECG_P_Onsets': 'precede',
    'ECG_P_Offsets': 'precede',
    'ECG_Q_Peaks': 'precede',
    'ECG_R_Onsets': 'precede',
    'ECG_R_Offsets': 'follow',
    'ECG_S_Peaks': 'follow',
    'ECG_T_Peaks': 'follow',
    'ECG_T_Onsets': 'follow',
    'ECG_T_Offsets': 'follow'
    }

    aligned_delineation_dict = {}
    for key, values in waves_dwt.items():
        category = category_mapping.get(key)

        if category:
            aligned_delineation = align_delineation_with_rpeaks(rpeaks, values, category=category)

            waves_dwt[key] = aligned_delineation

    p_wave_indices_good, qrs_indices_good, t_wave_indices_good = prepare_wave_indices_for_plotting(waves_dwt)
    if visualize == 'yes':
        print('good indices - after alignment function')
        visualize_ecg_with_waves(ecg_cleaned, p_wave_indices_good, qrs_indices_good,t_wave_indices_good)
        print('p_wave_indices_good after alingment',p_wave_indices_good )

    p_wave_indices, qrs_indices, t_wave_indices = filter_indices(waves_dwt)

    if visualize == 'yes':
        print('final indices - after filtering')
        visualize_ecg_with_waves(ecg_cleaned, p_wave_indices, qrs_indices,t_wave_indices)
        print('p_wave_indices before normalization',p_wave_indices)
    if visualize == 'yes':
        print('normalized signal and indices')
        visualize_ecg_with_waves(ecg_cleaned, p_wave_indices, qrs_indices,t_wave_indices)
        print('p_wave_indices after normalization',p_wave_indices)

    min_length = min(len(p_wave_indices), len(qrs_indices), len(t_wave_indices))

    filtered_p_wave_indices = []
    filtered_qrs_indices = []
    filtered_t_wave_indices = []

    for i in range(min_length):
        p_start, p_end = p_wave_indices[i]
        p_start, p_end = int(p_start), int(p_end)
        qrs_start, qrs_end = qrs_indices[i]
        qrs_start, qrs_end = int(qrs_start), int(qrs_end)
        t_start, t_end = t_wave_indices[i]
        t_start, t_end = int(t_start), int(t_end)

        filtered_p_wave_indices.append((p_start, p_end))
        filtered_qrs_indices.append((qrs_start, qrs_end))
        filtered_t_wave_indices.append((t_start, t_end))

    p_wave_data = [ecg_cleaned[start:end] for start, end in filtered_p_wave_indices]
    qrs_wave_data = [ecg_cleaned[start:end] for start, end in filtered_qrs_indices]
    t_wave_data = [ecg_cleaned[start:end] for start, end in filtered_t_wave_indices]

    if visualize == 'yes':
        print('final')
        visualize_ecg_with_waves(ecg_cleaned, filtered_p_wave_indices, filtered_qrs_indices, filtered_t_wave_indices)
        print('p_wave_indices final',filtered_p_wave_indices)
    return p_wave_data, qrs_wave_data, t_wave_data, filtered_p_wave_indices, filtered_qrs_indices, filtered_t_wave_indices

def get_wave_embeddings(extractor: WaveEmbeddingExtractor,
                        ecg_signal: np.ndarray,
                        p_wave_indices, qrs_indices, t_wave_indices):

    p_embs   = extractor.get_embeddings_roi_from_full(ecg_signal, p_wave_indices,
                                                      end_inclusive=True)
    qrs_embs = extractor.get_embeddings_roi_from_full(ecg_signal, qrs_indices,
                                                      end_inclusive=True)
    t_embs   = extractor.get_embeddings_roi_from_full(ecg_signal, t_wave_indices,
                                                      end_inclusive=True)

    p_list   = [e for e in p_embs]
    qrs_list = [e for e in qrs_embs]
    t_list   = [e for e in t_embs]
    return p_list, qrs_list, t_list

print("ECG records in total in this dataset: ", len(folder_names))

def has_empty_sublists(list_of_lists):
    return any(
        (isinstance(sublist, (list, np.ndarray)) and len(sublist) == 0) or sublist is None
        for sublist in list_of_lists
    )

patient_wave_data = {}
valid_patient_count = 0 
count = 0
which_lead = CONFIG["lead_index"]

os.makedirs(chunk_data_output_dir, exist_ok=True)
chunk_index = 0
start = time.time()

for file_index, folder_name in enumerate(folder_names, start=1):
    info = df_main.loc[df_main['filename'] == folder_name, 'label']
    data_label = info.iloc[0]

    data_key = f"{dataset_prefix}{folder_name}"
    obj = s3.get_object(Bucket=bucket_name, Key=data_key)['Body'].read()
    mat = scipy.io.loadmat(io.BytesIO(obj))
    sig = mat["val"]

    idx = which_lead
    ecg_signal = sig[idx, :]

    patient_id = f"{folder_name}" 
    count = count + 1
    try: 
        p_wave, qrs_wave, t_wave, p_wave_indices, qrs_indices, t_wave_indices = process_ecg_signal(ecg_signal, sampling_rate=sampling_rate)

        if not (p_wave and qrs_wave and t_wave and p_wave_indices and qrs_indices and t_wave_indices):
            print(f"Skipping ECG {patient_id} due to empty wave data or indices.")

            continue

        if any(len(wave) == 0 for wave in [p_wave, qrs_wave, t_wave]):
            print(f"Skipping ECG {patient_id} due to len = 0")

            continue

        skip_patient = False
        for index, sublist in enumerate(p_wave):
            if isinstance(sublist, (list, np.ndarray)) and len(sublist) == 0:
                print(f"Empty sublist found at index {index}: {sublist}")
                print(f"Skipping ECG {patient_id} due to empty sublist in p_wave.")
                skip_patient = True
                break

        if skip_patient:
            continue

        skip_patient = False
        for index, sublist in enumerate(qrs_wave):
            if isinstance(sublist, (list, np.ndarray)) and len(sublist) == 0:
                print(f"Empty sublist found at index {index}: {sublist}")
                print(f"Skipping ECG {patient_id} due to empty sublist in qrs_wave.")
                skip_patient = True
                break

        if skip_patient:
            continue

        skip_patient = False
        for index, sublist in enumerate(t_wave):
            if isinstance(sublist, (list, np.ndarray)) and len(sublist) == 0:
                print(f"Empty sublist found at index {index}: {sublist}")
                print(f"Skipping ECG {patient_id} due to empty sublist in t_wave.")
                skip_patient = True
                break

        if skip_patient:
            continue

        p_wave_features, qrs_wave_features, t_wave_features = get_wave_embeddings(
            extractor,
            np.asarray(ecg_signal, dtype=np.float32),
            p_wave_indices, qrs_indices, t_wave_indices
        )
        has_nan = False
        for feature_set in [p_wave_features, qrs_wave_features, t_wave_features]:
            for feature in feature_set:

                if torch.is_tensor(feature):
                    if torch.isnan(feature).any():
                        has_nan = True
                        break
                elif isinstance(feature, (list, np.ndarray)):
                    if np.isnan(np.array(feature)).any():
                        has_nan = True
                        break
            if has_nan:
                break

        if has_nan:
            print(f"Skipping patient {patient_id} due to NaN in wave features.")
            continue

        patient_wave_data[patient_id] = {
            "label": int(data_label), 
            'p_wave': p_wave,
            'qrs_wave': qrs_wave,
            't_wave': t_wave,
            'p_wave_indices': p_wave_indices,
            'qrs_indices': qrs_indices,
            't_wave_indices': t_wave_indices,
            'p_wave_features': p_wave_features,
            'qrs_wave_features': qrs_wave_features,
            't_wave_features': t_wave_features
        }
        valid_patient_count += 1

        if valid_patient_count % 100 == 0:
            chunk_file = os.path.join(chunk_data_output_dir, f"patient_wave_data_{chunk_index}.pkl")
            with open(chunk_file, 'wb') as f:
                pickle.dump(patient_wave_data, f)
            print(valid_patient_count)
            chunk_index += 1
            patient_wave_data = {}

    except Exception as e:
        print(f"Error: {patient_id} -> {type(e).__name__}: {e}")
        traceback.print_exc()
        continue

if len(patient_wave_data) > 0:
    chunk_file = os.path.join(chunk_data_output_dir, f"patient_wave_data_{chunk_index}.pkl")
    with open(chunk_file, 'wb') as f:
        pickle.dump(patient_wave_data, f)
    print(valid_patient_count)
end = time.time()
print('It took', (end - start) / 60, 'minutes')
print("Total valid patients processed:", valid_patient_count)

def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

patient_wave_data = {}
num_files = chunk_index
file_paths = [f"{chunk_data_output_dir}patient_wave_data_{i}.pkl" for i in range(0, num_files + 1)]

for file_path in file_paths:
    data_chunk = read_pickle_file(file_path)
    patient_wave_data.update(data_chunk)

print("Number of chunks:", num_files+1)
print("Total valid patients loaded:", len(patient_wave_data))

combined_p_wave_data = []
combined_qrs_wave_data = []
combined_t_wave_data = []

patient_indices = {}

for patient, waves in patient_wave_data.items():

    start_p = len(combined_p_wave_data)
    start_qrs = len(combined_qrs_wave_data)
    start_t = len(combined_t_wave_data)

    combined_p_wave_data.extend(waves['p_wave'])
    combined_qrs_wave_data.extend(waves['qrs_wave'])
    combined_t_wave_data.extend(waves['t_wave'])
    end_p = len(combined_p_wave_data)
    end_qrs = len(combined_qrs_wave_data)
    end_t = len(combined_t_wave_data)
    patient_indices[patient] = {
        'p_wave': (start_p, end_p),
        'qrs_wave': (start_qrs, end_qrs),
        't_wave': (start_t, end_t)
    }

print("Number of combined P wave entries:", len(combined_p_wave_data))
print("Number of combined QRS wave entries:", len(combined_qrs_wave_data))
print("Number of combined T wave entries:", len(combined_t_wave_data))

def analyze_wave_data(combined_wave_data):

    nan_count = 0
    none_count = 0
    empty_count = 0
    empty_sublists = []

    for index, sublist in enumerate(combined_wave_data):

        if isinstance(sublist, (list, np.ndarray)) and len(sublist) == 0:
            empty_count += 1
            empty_sublists.append((index, sublist))
        else:
            for value in sublist:
                if value is None:
                    none_count += 1
                elif isinstance(value, float) and np.isnan(value):
                    nan_count += 1

    summary = {
        "empty_count": empty_count,
        "none_count": none_count,
        "nan_count": nan_count,
    }

    return summary, empty_sublists

summary, empty_sublists = analyze_wave_data(combined_p_wave_data)

print(f"Summary: {summary}")
print("Empty Sublists (index and sublist):")
for idx, sublist in empty_sublists:
    print(f"Index: {idx}, Sublist: {sublist}")
summary, empty_sublists = analyze_wave_data(combined_qrs_wave_data)

print(f"Summary: {summary}")
print("Empty Sublists (index and sublist):")
for idx, sublist in empty_sublists:
    print(f"Index: {idx}, Sublist: {sublist}")
summary, empty_sublists = analyze_wave_data(combined_t_wave_data)

print(f"Summary: {summary}")
print("Empty Sublists (index and sublist):")
for idx, sublist in empty_sublists:
    print(f"Index: {idx}, Sublist: {sublist}")

def normalize_z_score(segment):
    if np.std(segment) == 0:
        return segment
    return (segment - np.mean(segment)) / np.std(segment)

combined_p_wave_data = [normalize_z_score(segment) for segment in combined_p_wave_data]
combined_qrs_wave_data = [normalize_z_score(segment) for segment in combined_qrs_wave_data]
combined_t_wave_data = [normalize_z_score(segment) for segment in combined_t_wave_data]
print("Wave data normalized using Z-score.")

def check_normalization(data, name):
    means = [np.mean(segment) for segment in data]
    stds = [np.std(segment) for segment in data]
    print(f"{name}: Mean (avg across segments) = {np.mean(means):.5f}, Std (avg across segments) = {np.mean(stds):.5f}")

check_normalization(combined_p_wave_data, "P-Wave")
check_normalization(combined_qrs_wave_data, "QRS-Wave")
check_normalization(combined_t_wave_data, "T-Wave")

for patient, indices in patient_indices.items():
    start_p, end_p = indices['p_wave']
    patient_wave_data[patient]['p_wave'] = combined_p_wave_data[start_p:end_p]
    start_qrs, end_qrs = indices['qrs_wave']
    patient_wave_data[patient]['qrs_wave'] = combined_qrs_wave_data[start_qrs:end_qrs]
    start_t, end_t = indices['t_wave']
    patient_wave_data[patient]['t_wave'] = combined_t_wave_data[start_t:end_t]

print("Patient wave data updated with normalized values.")

total_p_waves = sum(len(waves['p_wave']) for waves in patient_wave_data.values())
total_qrs_waves = sum(len(waves['qrs_wave']) for waves in patient_wave_data.values())
total_t_waves = sum(len(waves['t_wave']) for waves in patient_wave_data.values())
print("preprocessing of patient wave data is done! ")

max_length_p_wave = max(len(seq) for seq in combined_p_wave_data)  
max_length_qrs_wave = min(max(len(seq) for seq in combined_qrs_wave_data), 400)
max_length_t_wave = max(len(seq) for seq in combined_t_wave_data)

p_latent = encode_waves(combined_p_wave_data, p_ae_path, max_length_p_wave, 12, 128, DEVICE, 7).astype(np.float64)
qrs_latent = encode_waves(combined_qrs_wave_data, qrs_ae_path, max_length_qrs_wave, 24, 16, DEVICE, 9).astype(np.float64)
t_latent = encode_waves(combined_t_wave_data, t_ae_path, max_length_t_wave, 12, 128, DEVICE, 7).astype(np.float64)

p_latent = np.ascontiguousarray(np.asarray(p_latent, dtype=np.float64))
qrs_latent = np.ascontiguousarray(np.asarray(qrs_latent, dtype=np.float64))
t_latent = np.ascontiguousarray(np.asarray(t_latent, dtype=np.float64))

combined_p_wave_data_compressed = p_latent
combined_qrs_wave_data_compressed = qrs_latent
combined_t_wave_data_compressed = t_latent

from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np

def verify_normalization(data, data_name, tolerance=1e-6):
    global_mean = np.mean(data)
    global_std = np.std(data)

    print(f"\n{data_name} NORMALIZATION CHECK")
    print(f"Data shape: {data.shape}")
    print(f"Global mean: {global_mean:.6f}")
    print(f"Global std: {global_std:.6f}")
    print(f"Min value: {np.min(data):.6f}")
    print(f"Max value: {np.max(data):.6f}")

    if abs(global_mean) < tolerance:
        print("Mean is correctly normalized")
    else:
        print("Mean normalization failed")

    if abs(global_std - 1.0) < tolerance:
        print("Standard deviation is correctly normalized")
    else:
        print("Standard deviation normalization failed")

scaler_p = StandardScaler()
scaler_qrs = StandardScaler()
scaler_t = StandardScaler()

combined_p_wave_data_scaled = scaler_p.fit_transform(combined_p_wave_data_compressed)
combined_qrs_wave_data_scaled = scaler_qrs.fit_transform(combined_qrs_wave_data_compressed)
combined_t_wave_data_scaled = scaler_t.fit_transform(combined_t_wave_data_compressed)

verify_normalization(combined_p_wave_data_scaled, "P-WAVE")
verify_normalization(combined_qrs_wave_data_scaled, "QRS-WAVE")
verify_normalization(combined_t_wave_data_scaled, "T-WAVE")

normalizer_p = Normalizer(norm="l2")
normalizer_qrs = Normalizer(norm="l2")
normalizer_t = Normalizer(norm="l2")

combined_p_wave_data_scaled = normalizer_p.transform(combined_p_wave_data_scaled)
combined_qrs_wave_data_scaled = normalizer_qrs.transform(combined_qrs_wave_data_scaled)
combined_t_wave_data_scaled = normalizer_t.transform(combined_t_wave_data_scaled)

with open(p_cluster_path, 'rb') as f:   
    p_cluster_model = pickle.load(f)
with open(qrs_cluster_path,'rb') as f:  
    qrs_cluster_model = pickle.load(f)
with open(t_cluster_path, 'rb') as f:   
    t_cluster_model = pickle.load(f)

for _m in (p_cluster_model.get('model'), qrs_cluster_model.get('model'), t_cluster_model.get('model')):
    if hasattr(_m, 'cluster_centers_'):
        _m.cluster_centers_ = np.ascontiguousarray(_m.cluster_centers_.astype(np.float64, copy=False))

def _extract_centroids(obj):

    if isinstance(obj, dict) and "centroids" in obj:
        return np.asarray(obj["centroids"], dtype=np.float32)

    if isinstance(obj, dict) and "model" in obj:
        m = obj["model"]
        if hasattr(m, "cluster_centers_"):
            return np.asarray(m.cluster_centers_, dtype=np.float32)

    if isinstance(obj, (np.ndarray, list, tuple)):
        arr = np.asarray(obj, dtype=np.float32)
        if arr.ndim == 2:
            return arr
    raise ValueError("Unsupported clustering pickle format")

def load_centroids_flex(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return _extract_centroids(obj)

def predict_from_centroids(X, C, batch=65536):
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
    C_t = torch.as_tensor(np.asarray(C, dtype=np.float32))
    out = []
    for i in range(0, X_t.shape[0], batch):
        d = torch.cdist(X_t[i:i+batch], C_t)
        out.append(torch.argmin(d, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)

p_cent   = load_centroids_flex(p_cluster_path)
qrs_cent = load_centroids_flex(qrs_cluster_path)
t_cent   = load_centroids_flex(t_cluster_path)

p_latent = combined_p_wave_data_scaled
qrs_latent = combined_qrs_wave_data_scaled
t_latent = combined_t_wave_data_scaled

p_cluster_ids   = predict_from_centroids(p_latent,   p_cent)
qrs_cluster_ids = predict_from_centroids(qrs_latent, qrs_cent) + CONFIG["num_p_wave_clusters"]
t_cluster_ids   = predict_from_centroids(t_latent,   t_cent)   + CONFIG["num_p_wave_clusters"] + CONFIG["num_qrs_wave_clusters"]

for patient, idx in patient_indices.items():
    s_p,e_p = idx['p_wave'];  s_q,e_q = idx['qrs_wave'];  s_t,e_t = idx['t_wave']
    waves = patient_wave_data[patient]
    waves['p_cluster_ids'] = p_cluster_ids[s_p:e_p]
    waves['qrs_cluster_ids'] = qrs_cluster_ids[s_q:e_q]
    waves['t_cluster_ids'] = t_cluster_ids[s_t:e_t]

p_clusters_detected = sorted(np.unique(p_cluster_ids))
qrs_clusters_detected = sorted(np.unique(qrs_cluster_ids))
t_clusters_detected = sorted(np.unique(t_cluster_ids))

print(f"{len(p_clusters_detected)} unique P wave cluster IDs detected:", p_clusters_detected)
print(f"{len(qrs_clusters_detected)} unique QRS wave cluster IDs:", qrs_clusters_detected)
print(f"{len(t_clusters_detected)} unique T wave cluster IDs:", t_clusters_detected)

print("wave latent extraction and cluster ID assignment are done! ")

patient_beat_sentences = {}
patient_beat_sentences_with_cnn = {}
cnn_features_list = []

for patient, waves in patient_wave_data.items():
    p_cluster_ids = waves.get('p_cluster_ids', [])
    qrs_cluster_ids = waves.get('qrs_cluster_ids', [])
    t_cluster_ids = waves.get('t_cluster_ids', [])
    data_label = waves["label"]
    p_wave_features = waves.get('p_wave_features', [])
    qrs_wave_features = waves.get('qrs_wave_features', [])
    t_wave_features = waves.get('t_wave_features', [])

    num_sentences = min(len(p_cluster_ids), len(qrs_cluster_ids), len(t_cluster_ids))

    patient_beat_sentences[patient] = []
    patient_beat_sentences_with_cnn[patient] = []

    i = 0
    while i < num_sentences:

        num_beats = random.choice([1, 2, 6, 8])
        sentence = []
        cnn_sentence_features = []
        for _ in range(num_beats):
            if i >= num_sentences:
                break
            beat = [p_cluster_ids[i], qrs_cluster_ids[i], t_cluster_ids[i]]
            sentence.append(beat)
            cnn_sentence_features.append([p_wave_features[i], qrs_wave_features[i], t_wave_features[i]])
            i += 1
        if sentence:
            patient_beat_sentences[patient].append(sentence)
            patient_beat_sentences_with_cnn[patient].append(cnn_sentence_features)
    patient_beat_sentences[patient].append(data_label)
    patient_beat_sentences_with_cnn[patient].append(data_label)

print("An example of beat sentences:", patient_beat_sentences[list(patient_wave_data.keys())[0]])

patient_labels = [data["label"] for data in patient_wave_data.values()]
print(f"Label distribution:", Counter(patient_labels))

with open(fine_tuning_sentences_file_name, 'wb') as f:
    pickle.dump(patient_beat_sentences, f)

with open(fine_tuning_sentences_cnn_file_name, 'wb') as f:
    pickle.dump(patient_beat_sentences_with_cnn, f)
print("beat sentences are generated! ")
