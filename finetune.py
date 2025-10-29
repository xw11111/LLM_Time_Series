import json
import pickle
import random
import math
from typing import List, Tuple, Dict
import os
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pretrain import BERT
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
import argparse, sys

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import label_binarize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#-----------------------global parameters definition starts here-----------------------#                                  
CONFIG = {
    "dataset_name": "ptb",      # ptbxl/cpsc2018/cs
    "pretrained_folder": "pretrain",  # where the pre-training artefacts live
    "batch_size": 64,
    "num_epochs": 100,
    "lr": 1e-4,
    # "dropout": 0.2,
    "weight_decay": 1e-5,
    "patience": 10,
    'min_delta': 1e-6,
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["W_Q", "W_V", "W_K", "fc"],#, "fc1", "fc2"
    "official_split": False,
}

# ----------------------- CLI hyper-parameter overrides ---------------------
parser = argparse.ArgumentParser(description="Fine-tune parameters")
parser.add_argument("--dataset_name", type=str, help="dataset_name")
parser.add_argument("--pretrained_folder", type=str, help="pretrained_folder")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--lora_r", type=int, help="LoRA rank r")
parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
parser.add_argument("--weight_decay", type=float, help="weight decay")
parser.add_argument("--official_split", action="store_true", help="Use official train/val/test splits instead of random split")
parser.add_argument("--seed", type=int, default=0)
cli_args, _ = parser.parse_known_args()
for key in ["dataset_name", "pretrained_folder", "lr", "batch_size", "lora_r", "lora_alpha", "lora_dropout", "weight_decay"]:
    if getattr(cli_args, key) is not None:
        CONFIG[key] = getattr(cli_args, key)
# Handle boolean flag for official_split
if cli_args.official_split:
    CONFIG["official_split"] = True
    
def set_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = int(cli_args.seed)
set_seed(SEED)

# ---------------------------------------------------------------------------

suff = './pickles/new_pickles_800k_AE1/'                   

pretrained_path = suff + CONFIG["pretrained_folder"]
word_dict_path = pretrained_path + "/word_dict.pkl"
pretrain_params_path = pretrained_path + "/parameters.json" 
pretrain_model_path = pretrained_path + "/bert_model.pth"
finetune_path = suff + 'finetune/' + CONFIG["dataset_name"]
beat_sentence_pickle_name = finetune_path + '/beat_sentences_withID.pkl'
beat_sentence_cnn_pickle_name = finetune_path + '/beat_sentences_cnn_withID.pkl'
# fine_tune_output_path = os.path.join(finetune_path, "lora_adapters")
os.makedirs(finetune_path, exist_ok=True)
#-----------------------global parameters definition ends here-----------------------#

#-------------pre-training hyper-parameters & word_dict loading starts here--------------#
try: 
    with open(pretrain_params_path, "r") as f:
        pretrain_params = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("parameters.json not found. Please export it during pre-training.")

try:
    with open(word_dict_path, "rb") as f:
        word_dict = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("word_dict.pkl not found. Please export it during pre-training.")

max_seq_len = pretrain_params["max_seq_len"] # max length of one sentence
cnn_embed_dim = pretrain_params["cnn_embed_dim"]
vocab_size = pretrain_params["vocab_size"] 
#-------------pre-training hyper-parameters & word_dict loading ends here--------------#

#----------model definition (reuse architecture & add classifier) starts here-----------#
class BERTForClassification(nn.Module):
    """Wrap pre-trained BERT with a softmax classifier head."""

    def __init__(self, base_model: BERT, d_model: int, num_classes: int):
        super().__init__()
        self.bert = base_model
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, cnn_features, labels=None):
        pooled_output = self.bert(input_ids, cnn_features)  # [B, d_model]
        # self.dropout_head = nn.Dropout(CONFIG["dropout"]) #add a dropout before the classifier head
        # logits = self.classifier(self.dropout_head(pooled_output))
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return logits, loss
#----------model definition (reuse architecture & add classifier) ends here-----------#

#------------------------labeled dataset padding starts here--------------------------#
SEP_TOKEN = word_dict["[SEP]"]
PAD_TOKEN = word_dict["[PAD]"]

class ClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.max_seq_len = max_seq_len  #max length of a sentence

    def __len__(self):
        return len(self.df)

    def _process_sentence_tokens(self, sentence) -> torch.Tensor:
        cur_sentence = []
        for beat in sentence:
            for wave in beat:
                cur_sentence.append(word_dict.get(wave, PAD_TOKEN))
        cur_sentence.append(SEP_TOKEN)

        if len(cur_sentence) < self.max_seq_len:
            cur_sentence.extend([PAD_TOKEN] * (self.max_seq_len - len(cur_sentence)))
        return torch.tensor(cur_sentence[:self.max_seq_len], dtype=torch.long)

    def _process_sentence_cnn(self, cnn_sentence) -> torch.Tensor:
        zero_feat = torch.zeros(cnn_embed_dim, dtype=torch.float32)  # <- no (1, C)

        def _to_feat(feat):
            if feat is None:
                return zero_feat.clone()
            if isinstance(feat, np.ndarray):
                t = torch.from_numpy(feat.astype(np.float32, copy=False))
            elif torch.is_tensor(feat):
                t = feat.detach().to(torch.float32).cpu()
            else:
                t = torch.as_tensor(feat, dtype=torch.float32)

            t = t.reshape(-1)
            if t.numel() < cnn_embed_dim:
                t = torch.cat([t, torch.zeros(cnn_embed_dim - t.numel(), dtype=torch.float32)], dim=0)
            else:
                t = t[:cnn_embed_dim]
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            return t  # <- shape (C,)

        cur_cnn = []
        for beat in cnn_sentence:      # beat = [p, qrs, t]
            for feat in beat:
                cur_cnn.append(_to_feat(feat))

        cur_cnn.append(zero_feat.clone())           # SEP
        while len(cur_cnn) < self.max_seq_len:
            cur_cnn.append(zero_feat.clone())

        cur_cnn = cur_cnn[:self.max_seq_len]
        return torch.stack(cur_cnn, dim=0)  # [max_seq_len, cnn_embed_dim]


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lab_raw = row["label"]
        is_one_based = CONFIG["dataset_name"] in ["ptb", "cpsc2018"]
        label_int = int(lab_raw) - (1 if is_one_based else 0)

        token_ids    = self._process_sentence_tokens(row["sentence"])
        cnn_features = self._process_sentence_cnn(row["cnn_sentence"])

        return {
            "input_ids":    token_ids,
            "cnn_features": cnn_features,
            "labels":       torch.tensor(label_int, dtype=torch.long),
        }
#------------------------labeled dataset padding ends here--------------------------#

#------------------labeled dataset loading starts here------------------#
with open(beat_sentence_pickle_name, "rb") as f:
    patient_sent_dict = pickle.load(f)  # {patient_id: ([sents], label)}

with open(beat_sentence_cnn_pickle_name, "rb") as f:
    patient_sent_cnn_dict = pickle.load(f)  # {patient_id: ([cnn_sents], label)}

# ---------------------- helper: read official split maps ------------------- #
def load_split_map(ds: str) -> Dict[str, str]:
    """Return dict {patient_id: split} where split in {train,val,test}."""
    try:
        import boto3
        s3 = boto3.client("s3")
        BUCKET = "walkky-datasets"
        
        def read_csv_from_s3(key: str) -> pd.DataFrame:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except ImportError:
        print("Warning: boto3 not available. Cannot use official splits for PTB-XL/CPSC2018.")
        return {}
    except Exception as e:
        print(f"Warning: Could not connect to S3: {e}")
        return {}
    
    ds = ds.lower()
    if ds == "ptb" or ds == "ptbxl":
        try:
            df = read_csv_from_s3("data/ptbxl/ptbxl_database.csv")
            def fold_to_split(f):
                f = int(f)
                if 1<=f<=8: return "train"
                if f==9: return "val"
                if f==10: return "test"
                return None
            df["split"] = df["strat_fold"].apply(fold_to_split)
            return {f"ECG_{int(eid):05d}": sp for eid,sp in zip(df.ecg_id, df.split) if sp}
        except Exception as e:
            print(f"Warning: Could not load PTB-XL split map: {e}")
            return {}
            
    if ds == "cpsc2018":
        try:
            split_files = {
                "train" : "data/cpsc_2018/icbeb_train.csv",
                "val" : "data/cpsc_2018/icbeb_val.csv",
                "test" : "data/cpsc_2018/icbeb_test.csv",
            }
            mapping = {}
            for sp,key in split_files.items():
                ids = read_csv_from_s3(key)["filename"].apply(lambda p: os.path.basename(str(p)))
                for fn in ids: mapping[fn] = sp
            return mapping
        except Exception as e:
            print(f"Warning: Could not load CPSC2018 split map: {e}")
            return {}
            
    if ds == "cs":
        # no official split
        return {}
    
    print(f"Warning: Unsupported dataset_name for official splits: {ds}")
    return {}

# ---------------------- build train/val/test patient lists ----------------- #
all_patients = list(patient_sent_dict.keys())

if CONFIG["official_split"]:
    print(f"Using official splits for dataset: {CONFIG['dataset_name']}")
    split_map = load_split_map(CONFIG["dataset_name"])
    
    if not split_map and CONFIG["dataset_name"] != "cs":
        print("Warning: Official split map is empty, falling back to random split")
        CONFIG["official_split"] = False
    else:
        if CONFIG["dataset_name"] == "cs":
            print("Chapman-Shaffer dataset has no official split, using random split")
            # For CS dataset, use random split even if official_split is True
            random.seed(SEED)
            random.shuffle(all_patients)
            n = len(all_patients)
            train_patients = set(all_patients[: int(0.7*n)])
            val_patients = set(all_patients[int(0.7*n): int(0.8*n)])
            test_patients = set(all_patients[int(0.8*n):])
        else:
            # use official mapping (PTB-XL / CPSC2018)
            train_patients = {pid for pid,sp in split_map.items() if sp=="train"}
            val_patients = {pid for pid,sp in split_map.items() if sp=="val"}
            test_patients = {pid for pid,sp in split_map.items() if sp=="test"}
            # keep only patients we actually have in our pickles
            train_patients &= set(all_patients)
            val_patients &= set(all_patients)
            test_patients &= set(all_patients)

if not CONFIG["official_split"]:
    print("Using random patient-level split (70/10/20)")
    # Original random split logic
    random.seed(SEED)
    unique_patients = list(patient_sent_dict.keys())
    random.shuffle(unique_patients)

    train_end = int(0.7 * len(unique_patients))
    val_end = int(0.8 * len(unique_patients))

    train_patients = set(unique_patients[:train_end])
    val_patients = set(unique_patients[train_end:val_end])
    test_patients = set(unique_patients[val_end:])

print(f"Patient counts | train: {len(train_patients)}, val: {len(val_patients)}, test: {len(test_patients)}")

# ---------------------- convert to sentence-level DataFrame --------------- #
rows: List[dict] = []
for pid in all_patients:
    split = (
        "train" if pid in train_patients else
        "val" if pid in val_patients else
        "test" if pid in test_patients else None
    )
    if split is None:
        continue  # patient not in any split (probably missing official label file)
    
    token_rec = patient_sent_dict[pid]
    cnn_rec = patient_sent_cnn_dict[pid]
    sentences, cnn_sentences = token_rec[:-1], cnn_rec[:-1]
    label = token_rec[-1]   # now numeric already
    
    assert len(sentences) == len(cnn_sentences), f"Token/CNN sentence length mismatch for patient {pid}"
    
    for s, c in zip(sentences, cnn_sentences):
        rows.append({
            "patient_id": pid,
            "sentence": s,         # one sentence (list of beats)
            "cnn_sentence": c,     # one cnn sentence (list of beat features)
            "label": label,
            "split": split,
        })

labeled_df = pd.DataFrame(rows)
print("Total number of sentence inputs:", len(labeled_df))
print("Sentence counts by split:")
print(labeled_df["split"].value_counts())

def is_invalid_cnn_sentence(cnn_sentence) -> bool:
    # True if any feature is None, empty, or has non-finite numbers
    for beat in cnn_sentence:
        if not isinstance(beat, (list, tuple)) or len(beat) != 3:
            return True
        for feat in beat:
            if feat is None:
                return True
            arr = np.asarray(feat)
            if arr.size == 0:            # empty sublist → invalid
                return True
            if not np.isfinite(arr).all():  # has NaN/Inf
                return True
    return False

bad_mask = labeled_df["cnn_sentence"].map(is_invalid_cnn_sentence)
n_bad = int(bad_mask.sum())
if n_bad > 0:
    print(f"Dropping {n_bad} / {len(labeled_df)} sentences with empty/NaN CNN features.")
    labeled_df = labeled_df.loc[~bad_mask].reset_index(drop=True)


# Filter DataFrames by split
train_df = labeled_df[labeled_df['split'] == 'train']
val_df = labeled_df[labeled_df['split'] == 'val']
test_df = labeled_df[labeled_df['split'] == 'test']

train_dataset = ClassificationDataset(train_df)
val_dataset = ClassificationDataset(val_df)
test_dataset = ClassificationDataset(test_df)

def seed_worker(worker_id):
    # Each worker gets a different, but deterministic seed
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset,batch_size=CONFIG["batch_size"],shuffle=True,drop_last=True,worker_init_fn=seed_worker,generator=g)
val_loader = DataLoader(val_dataset,batch_size=CONFIG["batch_size"],shuffle=False,worker_init_fn=seed_worker,generator=g)
test_loader = DataLoader(test_dataset,batch_size=CONFIG["batch_size"],shuffle=False,worker_init_fn=seed_worker,generator=g)


def evaluate_on_loader(model, loader, num_classes: int):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            cnn_feats  = batch["cnn_features"].to(device)
            labels     = batch["labels"].to(device)

            logits, loss = model(input_ids, cnn_feats, labels=labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)

    # overall metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    if CONFIG["dataset_name"] == "cs":
        # macro AUROC (robust if some classes absent)
        valid_aucs = []
        valid_classes = []
        for c in range(num_classes):
            y_true_c = (all_labels == c).astype(int)
            pos = y_true_c.sum()
            neg = len(y_true_c) - pos
            if pos == 0 or neg == 0:
                continue  # cannot compute AUC for this class in this split
            try:
                auc_c = roc_auc_score(y_true_c, all_probs[:, c])
                valid_aucs.append(auc_c)
                valid_classes.append(c)
            except ValueError:
                pass

        macro_auroc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else float("nan")
        # Optional: quick visibility into what's happening
        if len(valid_aucs) == 0:
            print("Warning: no valid classes for AUC in this split (all missing or single-class).")
        else:
            pass
    else:
        # macro AUROC (one-vs-rest)
        y_true_onehot = label_binarize(all_labels, classes=list(range(num_classes)))
        try:
            macro_auroc = roc_auc_score(y_true_onehot, all_probs, average="macro", multi_class="ovr")
        except ValueError:
            macro_auroc = float("nan")

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, acc, macro_f1, macro_auroc

#------------------labeled dataset loading & splitting ends here------------------#

def infer_offset_and_nclasses(df_labels: pd.Series) -> tuple[int, int]:
    arr = df_labels.astype(int).to_numpy()
    lbl_min, lbl_max = int(arr.min()), int(arr.max())
    # 1 to k if starts from 0 no shift
    offset = 0 if lbl_min == 0 else 1
    n_classes = lbl_max - offset + 1
    return offset, int(n_classes)

LABEL_OFFSET, num_classes = infer_offset_and_nclasses(labeled_df["label"])

#------------------------BERT & LoRA adapters starts here------------------------#
base_bert = BERT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=pretrain_params["d_model"],
    n_layers=pretrain_params["n_layers"],
    n_heads=pretrain_params["n_heads"],
    d_k=pretrain_params["d_k"],
    d_v=pretrain_params["d_v"],
    d_ff=pretrain_params["d_ff"],
    cnn_embed_dim=cnn_embed_dim,
)
base_bert.load_state_dict(torch.load(pretrain_model_path, map_location=device), strict=True) 
# if confident about the model weights for bert base model, use True to ensure all keys match, if contains extra weights like from pretraining MLM head, use False then

model = BERTForClassification(base_bert, pretrain_params["d_model"], num_classes)

# LoRA
lora_cfg = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=CONFIG["target_modules"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
)
model = get_peft_model(model, lora_cfg)

# sanity-check trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable / 1e6:.2f}M / {all_params / 1e6:.2f}M")

model.to(device)
#------------------------BERT & LoRA adapters ends here------------------------#

#---------------------------model training starts here---------------------------#
optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

# add a linear warm-up for first 10% of steps
total_steps  = len(train_loader) * CONFIG["num_epochs"]
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
) 

# metrics
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []
patience = CONFIG["patience"]
epochs_no_improve = 0

test_losses, test_accs, test_macro_f1s, test_macro_aurocs = [], [], [], []

for epoch in range(1, CONFIG["num_epochs"] + 1):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        cnn_feats = batch["cnn_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, loss = model(input_ids, cnn_feats, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step() #learning warm-up added

        running_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    avg_train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train if total_train > 0 else 0.0

    # evaluation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            cnn_feats = batch["cnn_features"].to(device)
            labels = batch["labels"].to(device)
            logits, loss = model(input_ids, cnn_feats, labels=labels)
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)
    t_loss, t_acc, t_f1, t_auc = evaluate_on_loader(model, test_loader, num_classes)
    test_losses.append(t_loss); test_accs.append(t_acc)
    test_macro_f1s.append(t_f1); test_macro_aurocs.append(t_auc)

    print(f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | test_macroAUC={t_auc:.4f}")

    # store metrics for plotting
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_acc > best_val_acc + CONFIG["min_delta"]:
        best_val_acc = val_acc
        model.save_pretrained(finetune_path)
        print("Saved new best LoRA adapters.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered – no val_acc improvement for", patience, "epochs")
            break

print("Training finished. Best val acc =", best_val_acc)

# plot curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="train_loss")
plt.plot(val_losses, label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss curves"); plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="train_acc")
plt.plot(val_accs, label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy curves"); plt.legend()

plt.tight_layout()
plot_path = os.path.join(finetune_path, "training_curves.png")
plt.savefig(plot_path)
print("Saved training curves to", plot_path)
#---------------------------model training ends here---------------------------#

best_epoch = int(np.nanargmax(test_macro_aurocs)) + 1
print(f"Best test Macro-AUC epoch: {best_epoch} "
      f"(AUC={test_macro_aurocs[best_epoch-1]:.4f})")


# #---------------------------model evaluation on test set starts here---------------------------#
# print("\n--- Evaluating on Test Set ---")
# print("Loading best model for testing...")
# '''
# safetensor_path = os.path.join(finetune_path, "adapter_model.safetensors")
# adapter_state = load_file(safetensor_path)
# model.load_state_dict(adapter_state, strict=False)
# print("Loaded adapters from adapter_model.safetensors")

# model.eval()
# '''
# base_bert = BERT(
#     vocab_size=vocab_size,
#     max_seq_len=max_seq_len,
#     d_model=pretrain_params["d_model"],
#     n_layers=pretrain_params["n_layers"],
#     n_heads=pretrain_params["n_heads"],
#     d_k=pretrain_params["d_k"],
#     d_v=pretrain_params["d_v"],
#     d_ff=pretrain_params["d_ff"],
#     cnn_embed_dim=cnn_embed_dim,
# )
# base_bert.load_state_dict(torch.load(pretrain_model_path, map_location=device), strict=True)

# model = BERTForClassification(base_bert, pretrain_params["d_model"], num_classes)

# # 2) Wrap with the same LoRA config
# lora_cfg = LoraConfig(
#     r=CONFIG["lora_r"],
#     lora_alpha=CONFIG["lora_alpha"],
#     target_modules=CONFIG["target_modules"],
#     lora_dropout=CONFIG["lora_dropout"],
#     bias="none",
# )
# model = get_peft_model(model, lora_cfg)

# # 3) Load the best adapters that you saved with `model.save_pretrained(finetune_path)`
# model = PeftModel.from_pretrained(model, finetune_path)
# model.to(device)
# model.eval()
# correct = 0
# total = 0
# test_loss = 0.0
# all_labels = []
# all_preds = []
# all_prob_vecs = []  # store probability vector per sample

# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch["input_ids"].to(device)
#         cnn_feats = batch["cnn_features"].to(device)
#         labels = batch["labels"].to(device)
#         logits, loss = model(input_ids, cnn_feats, labels=labels)
#         test_loss += loss.item()
        
#         probs_vec = torch.softmax(logits, dim=1)
#         preds = probs_vec.argmax(dim=1)

#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#         all_labels.extend(labels.cpu().numpy())
#         all_preds.extend(preds.cpu().numpy())
#         all_prob_vecs.extend(probs_vec.detach().cpu().numpy())

# from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
# from sklearn.preprocessing import label_binarize

# # overall metrics
# test_accuracy = accuracy_score(all_labels, all_preds)
# macro_f1 = f1_score(all_labels, all_preds, average='macro')

# all_prob_vecs_np = np.vstack(all_prob_vecs)
# y_true_onehot = label_binarize(all_labels, classes=list(range(num_classes)))

# try:
#     macro_auroc = roc_auc_score(y_true_onehot, all_prob_vecs_np, average='macro', multi_class='ovr')
# except ValueError:
#     macro_auroc = float('nan')

# print(f"\nTest Loss: {test_loss/len(test_loader):.4f} | Accuracy: {test_accuracy:.4f} | Macro-F1: {macro_f1:.4f} | Macro-AUROC: {macro_auroc:.4f}")

# # Per-class metrics
# prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
# auroc_arr = []
# for c in range(num_classes):
#     try:
#         auroc_c = roc_auc_score(y_true_onehot[:, c], all_prob_vecs_np[:, c])
#     except ValueError:
#         auroc_c = float('nan')
#     auroc_arr.append(auroc_c)

# print("\nPer-class metrics:")
# print("Cls | Precision Recall  F1  AUROC")
# for c in range(num_classes):
#     print(f" {c:2d} |   {prec_arr[c]:.3f}    {rec_arr[c]:.3f}  {f1_arr[c]:.3f}  {auroc_arr[c]:.3f}")

# #----------------------------model evaluation on test set ends here----------------------------#