#!/usr/bin/env python3
import boto3
import pickle
import pandas as pd
import numpy as np
import time
import random
from random import shuffle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import math
import json
import os
import warnings
warnings.simplefilter("ignore")

# ================== DDP imports & helpers ==================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def ddp_setup():
    """Initialize DDP if launched with torchrun; return (rank, world_size, local_rank, device)."""
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        # Single-process fallback
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device

def is_main_process():
    return (not ddp_is_initialized()) or dist.get_rank() == 0

def ddp_allreduce_scalar(x, device):
    """All-reduce a python float across ranks (SUM) and return the reduced float."""
    if not ddp_is_initialized():
        return x
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()

# ============================================================

suff = './pickles/new_pickles_800k_AE1/pretrain/'
suff2 = './pickles/new_pickles_800k_AE1/'

beat_sentence_pickle_name = 'beat_sentences_withID.pkl'
beat_sentence_cnn_pickle_name = 'beat_sentences_cnn_withID.pkl'
train_results_filename = suff + 'train_results.csv'
bert_model_filename = suff + 'bert_model.pth'
train_plot_filename = suff + 'train_plot.png'
parameters_filename = suff + 'parameters.json'
word_dict_filename = suff + 'word_dict.pkl'
os.makedirs(suff, exist_ok=True)

CONFIG = {
    'cnn_embed_dim': 512,
    'batch_size': 64,
    'n_layers': 8,
    'n_heads': 12,
    'd_model': 192,
    'num_epoch': 100,
    'patience': 10,
    'min_delta': 1e-5,
    'min_lr': 1e-5,
    'max_lr': 1e-4,
    'mask_ratio': 0.2,
    'weight_decay': 3e-4,
}

# ------------------- CLI overrides -------------------
import argparse, sys
parser = argparse.ArgumentParser(description="BERT pretraining arguments")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--n_layers", type=int)
parser.add_argument("--n_heads", type=int)
parser.add_argument("--d_model", type=int)
parser.add_argument("--max_lr", type=float)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--mask_ratio", type=float)
parser.add_argument("--seed", type=int, default=42)

cli_args, _ = parser.parse_known_args()
for k in ["batch_size","n_layers","n_heads","d_model","max_lr","weight_decay","mask_ratio"]:
    if getattr(cli_args, k) is not None:
        CONFIG[k] = getattr(cli_args, k)

# ================== DDP init ==================
rank, world_size, local_rank, device = ddp_setup()
if is_main_process():
    print(f"Using device: {device}  | rank {rank}/{world_size-1}  | local_rank={local_rank}")

def set_seed(seed: int):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_attn_mask.to(seq_q.device)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        residual = inputs
        output = self.relu(self.fc1(inputs))
        output = self.dropout(output)
        output = self.fc2(output)
        return self.norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Embedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, cnn_embed_dim):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.cnn_linear = nn.Linear(cnn_embed_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, cnn_features):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        tok_embedding = self.tok_embed(x)
        pos_embedding = self.pos_embed(pos)
        cnn_embedding = self.cnn_linear(cnn_features)
        combined = tok_embedding + pos_embedding + cnn_embedding
        return self.norm(combined)

class BERT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, n_heads, d_k, d_v, d_ff, cnn_embed_dim, dropout=0.12):
        super().__init__()
        self.embedding = Embedding(vocab_size, max_seq_len, d_model, cnn_embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
    def forward(self, input_ids, cnn_features, masked_pos=None):
        output = self.embedding(input_ids, cnn_features)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        mask = (input_ids != 0).unsqueeze(-1).float()
        masked_output = output * mask
        h_pooled = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        h_pooled = self.activ1(self.fc(h_pooled))
        h_pooled = self.dropout(h_pooled)
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(self.activ2(self.linear(h_masked)))
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            return logits_lm
        return h_pooled

# ----- dataset & collate -----
class ECGSentenceDataset(Dataset):
    def __init__(self, tokens_list, cnn_list):
        assert len(tokens_list) == len(cnn_list)
        self.tokens = tokens_list
        self.cnn = cnn_list
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        return self.tokens[idx], self.cnn[idx]

def seed_worker(worker_id):
    import numpy as np
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

def make_collate_fn(max_seq_len, max_pred_unused, mask_ratio, cnn_embed_dim):
    PAD, SEP, MASK = 0, 1, 2

    def _to_vec1d(x):
        t = (x.detach().cpu().to(torch.float32) if torch.is_tensor(x)
             else torch.as_tensor(x, dtype=torch.float32))
        if t.ndim == 2 and t.size(0) == 1: t = t.squeeze(0)
        if t.ndim != 1: t = t.reshape(-1)
        if t.numel() < cnn_embed_dim:
            t = torch.cat([t, torch.zeros(cnn_embed_dim - t.numel(), dtype=torch.float32)])
        elif t.numel() > cnn_embed_dim:
            t = t[:cnn_embed_dim]
        return t

    def collate(batch):
        per_ids, per_mtok, per_mpos, per_feats = [], [], [], []

        for tokens_a, cnn_feats_a in batch:
            tokens = tokens_a[:max_seq_len]
            if len(tokens) < max_seq_len:
                tokens = tokens + [PAD] * (max_seq_len - len(tokens))

            feats = []
            for beat in cnn_feats_a[:max_seq_len]:
                if isinstance(beat, list):
                    feats.extend([_to_vec1d(b) for b in beat])
                else:
                    feats.append(_to_vec1d(beat))

            # pad with zeros (C,) to max_seq_len
            pad_vec = torch.zeros(cnn_embed_dim, dtype=torch.float32)
            if len(feats) == 0:
                feats.append(pad_vec.clone())
            while len(feats) < max_seq_len:
                feats.append(pad_vec.clone())
            feats = feats[:max_seq_len]

            # dynamic masking
            ids = list(tokens)
            candidates = [j for j, t in enumerate(ids) if t not in (PAD, SEP)]
            n_pred = max(1, int(round(len(candidates) * mask_ratio)))
            random.shuffle(candidates)
            m_pos, m_tok = [], []
            for pos in candidates[:n_pred]:
                m_pos.append(pos)
                m_tok.append(ids[pos])
                ids[pos] = MASK

            per_ids.append(torch.tensor(ids, dtype=torch.long))
            per_mtok.append(torch.tensor(m_tok, dtype=torch.long))
            per_mpos.append(torch.tensor(m_pos, dtype=torch.long))
            per_feats.append(torch.stack(feats, dim=0))

        import torch.nn.utils.rnn as rnn_utils
        if len(per_mpos) == 0:
            per_mtok = [torch.zeros(1, dtype=torch.long)]
            per_mpos = [torch.zeros(1, dtype=torch.long)]

        masked_tokens = rnn_utils.pad_sequence(per_mtok, batch_first=True, padding_value=0)
        masked_pos    = rnn_utils.pad_sequence(per_mpos, batch_first=True, padding_value=0)
        input_ids     = torch.stack(per_ids,  dim=0)
        cnn_features  = torch.stack(per_feats, dim=0)

        return input_ids, masked_tokens, masked_pos, cnn_features

    return collate


def calculate_mlm_accuracy_counts(logits_lm, masked_tokens):
    """Return (#correct, #valid) so we can aggregate across DDP ranks properly."""
    pred_tokens = logits_lm.argmax(dim=2)
    valid_mask = masked_tokens.ne(0)
    correct = (pred_tokens == masked_tokens) & valid_mask
    return correct.sum().item(), valid_mask.sum().item()

def evaluate_bert_loader(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    acc_correct = 0
    acc_total = 0
    num_batches = 0
    with torch.no_grad():
        for input_ids, masked_tokens, masked_pos, cnn_features in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            masked_tokens = masked_tokens.to(device, non_blocking=True)
            masked_pos = masked_pos.to(device, non_blocking=True)
            cnn_features = cnn_features.to(device, non_blocking=True)
            logits_lm = model(input_ids, cnn_features, masked_pos)
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens).mean()
            correct, total = calculate_mlm_accuracy_counts(logits_lm, masked_tokens)
            loss_sum += loss_lm.item()
            acc_correct += correct
            acc_total += total
            num_batches += 1

    # ---- DDP: aggregate sums across ranks ----
    if ddp_is_initialized():
        loss_sum = ddp_allreduce_scalar(loss_sum, device)
        num_batches = ddp_allreduce_scalar(float(num_batches), device)
        acc_correct = ddp_allreduce_scalar(float(acc_correct), device)
        acc_total = ddp_allreduce_scalar(float(acc_total), device)

    loss_avg = (loss_sum / max(1.0, num_batches))
    acc_avg = (acc_correct / max(1.0, acc_total))
    return loss_avg, acc_avg

def main():
    set_seed(cli_args.seed)

    with open(suff2 + beat_sentence_pickle_name, 'rb') as f:
        patient_beat_sentences = pickle.load(f)
    if is_main_process():
        print("Number of data samples for pretraining BERT:", len(patient_beat_sentences))

    with open(suff2 + beat_sentence_cnn_pickle_name, 'rb') as f:
        patient_beat_sentences_with_cnn = pickle.load(f)
    if len(patient_beat_sentences_with_cnn) != len(patient_beat_sentences):
        print("Error: Number of data samples in patient_beat_sentences_with_cnn is not equal to patient_beat_sentences!")

    SEP_TOKEN = '[SEP]'
    PAD_TOKEN = '[PAD]'
    MASK_TOKEN = '[MASK]'
    word_dict = {PAD_TOKEN: 0, SEP_TOKEN: 1, MASK_TOKEN: 2}
    current_index = len(word_dict)

    def create_word_dict(patient_beat_sentences, start_index=current_index):
        max_seq_len = 0
        global num_inputs
        num_inputs = 0
        wave_cluster_indices = set()
        for sentences in patient_beat_sentences.values():
            for sentence in sentences:
                for beat in sentence:
                    for wave in beat:
                        wave_cluster_indices.add(wave)
                seq_len = len(sentence) * 3 + 1
                max_seq_len = max(max_seq_len, seq_len)
                num_inputs += 1
        if is_main_process():
            print("Number of inputs (sentences in total):", num_inputs)
        wave_cluster_indices = sorted(wave_cluster_indices)
        for item in wave_cluster_indices:
            word_dict[item] = start_index
            start_index += 1
        return max_seq_len, word_dict

    max_seq_len, word_dict = create_word_dict(patient_beat_sentences)
    if is_main_process():
        print("Maximum number of tokens in a sentence (max_seq_len):", max_seq_len)
        print("Word Dictionary:", word_dict)

    if is_main_process():
        with open(word_dict_filename, 'wb') as f:
            pickle.dump(word_dict, f)
        print(f"Word dictionary saved to {word_dict_filename}")
        print("Volcabulary size (number of unique tokens):", len(word_dict))
    vocab_size = len(word_dict)

    # ----- split with seed (consistent across ranks) -----
    patient_ids = list(patient_beat_sentences.keys())
    seed = cli_args.seed
    rng_split = random.Random(seed)
    rng_split.shuffle(patient_ids)
    split_index1 = int(len(patient_ids) * 0.9)
    train_patients = patient_ids[:split_index1]
    val_patients = patient_ids[split_index1:]
    train_set = {patient: patient_beat_sentences[patient] for patient in train_patients}
    val_set = {patient: patient_beat_sentences[patient] for patient in val_patients}
    if is_main_process():
        print("Number of records in training set:", len(train_set.keys()))
        print("First patient's beat sentences in training set:", train_set[list(train_set.keys())[0]])

    train_set_with_cnn = {patient: patient_beat_sentences_with_cnn[patient] for patient in train_patients}
    val_set_with_cnn = {patient: patient_beat_sentences_with_cnn[patient] for patient in val_patients}
    if len(train_set.keys()) != len(train_set_with_cnn.keys()):
        print("Error: Number of patients in training set is not equal to number of patients in training set with CNN!")

    def process_patient_beat_sentences_with_cnn(patient_beat_sentences, patient_beat_sentences_with_cnn, max_len=max_seq_len):
        def to_cpu_tensor(x):
            if torch.is_tensor(x):
                return x.detach().to("cpu")
            return torch.as_tensor(x, dtype=torch.float32, device="cpu")

        processed_sentences = {}
        processed_cnn_features = {}

        for patient, sentences in patient_beat_sentences.items():
            processed_sentences[patient] = []
            for sentence in sentences:
                cur_sentence = []
                for beat in sentence:
                    cur_sentence.extend([word_dict[wave] for wave in beat])
                cur_sentence.append(word_dict[SEP_TOKEN])
                while len(cur_sentence) < max_len:
                    cur_sentence.append(word_dict[PAD_TOKEN])
                cur_sentence = cur_sentence[:max_len]
                processed_sentences[patient].append(cur_sentence)

        for patient, cnn_sentences in patient_beat_sentences_with_cnn.items():
            processed_cnn_features[patient] = []
            for cnn_sentence in cnn_sentences:
                cur_sentence = []
                for beat in cnn_sentence:
                    if isinstance(beat, list):
                        cur_sentence.extend([to_cpu_tensor(b) for b in beat])
                    else:
                        cur_sentence.append(to_cpu_tensor(beat))
                if cur_sentence:
                    zero_like = torch.zeros_like(cur_sentence[0], device="cpu")
                    cur_sentence.append(zero_like)  # [SEP] as zeros
                while len(cur_sentence) < max_len:
                    cur_sentence.append(torch.zeros_like(cur_sentence[0], device="cpu"))  # [PAD] as zeros
                cur_sentence = cur_sentence[:max_len]
                processed_cnn_features[patient].append(cur_sentence)

        return processed_sentences, processed_cnn_features

    def remove_patient_ids(data):
        result = []
        for value in data.values():
            result.extend(value)
        return result

    def create_token_patient_beat_sentences(patient_beat_sentences, patient_beat_sentences_with_cnn):
        processed_sentences, processed_cnn_features = process_patient_beat_sentences_with_cnn(
            patient_beat_sentences, patient_beat_sentences_with_cnn
        )
        processed_sentences_list = remove_patient_ids(processed_sentences)
        processed_cnn_features_list = remove_patient_ids(processed_cnn_features)
        return processed_sentences_list, processed_cnn_features_list

    train_token_list, train_token_list_cnn = create_token_patient_beat_sentences(train_set, train_set_with_cnn)
    val_token_list, val_token_list_cnn = create_token_patient_beat_sentences(val_set, val_set_with_cnn)
    if is_main_process():
        print('first training token list', train_token_list[0])
    del train_set, val_set, train_set_with_cnn, val_set_with_cnn
    del patient_beat_sentences, patient_beat_sentences_with_cnn

    batch_size = CONFIG["batch_size"]           # per-process (per-GPU) batch size
    n_layers = CONFIG["n_layers"]
    n_heads = CONFIG["n_heads"]
    d_model = CONFIG["d_model"]
    d_ff = d_model * 4
    d_k = d_v = d_model // n_heads
    cnn_embed_dim = CONFIG["cnn_embed_dim"]
    mask_ratio = CONFIG["mask_ratio"]

    parameters = {
        'num_inputs': num_inputs, 'num_val_inputs': len(val_token_list),
        'max_seq_len': max_seq_len, 'batch_size': batch_size,
        'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model,
        'd_ff': d_ff, 'd_k': d_k, 'd_v': d_v, 'vocab_size': vocab_size, 'cnn_embed_dim': cnn_embed_dim,
        'mask_ratio': mask_ratio, 'weight_decay': CONFIG["weight_decay"],
    }
    if is_main_process():
        with open(parameters_filename, "w") as f:
            json.dump(parameters, f, indent=4)

    train_ds = ECGSentenceDataset(train_token_list, train_token_list_cnn)
    val_ds  = ECGSentenceDataset(val_token_list,  val_token_list_cnn)
    collate_fn = make_collate_fn(max_seq_len, None, mask_ratio, cnn_embed_dim)

    # ================== DDP samplers & loaders ==================
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False) if ddp_is_initialized() else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if ddp_is_initialized() else None

    # Note: when using a sampler, set shuffle=False in DataLoader.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False if train_sampler is not None else True,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )

    num_epoch = CONFIG["num_epoch"]
    train_losses, train_mlm_accuracies = [], []
    val_losses, val_mlm_accuracies = [], []

    model = BERT(vocab_size, max_seq_len, d_model, n_layers, n_heads, d_k, d_v, d_ff, cnn_embed_dim).to(device)
    # Wrap with DDP
    if ddp_is_initialized():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
    optimizer = AdamW(model.parameters(), lr=CONFIG["max_lr"], weight_decay=CONFIG["weight_decay"], betas=(0.9, 0.99))

    total_steps = len(train_loader) * num_epoch
    warmup_steps = int(0.1 * total_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    patience = CONFIG["patience"]
    min_delta = CONFIG["min_delta"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(num_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # IMPORTANT for proper shuffling across epochs

        if is_main_process():
            print(f'Epoch: {epoch+1}/{num_epoch}')
        model.train()
        epoch_loss_local = 0.0
        mlm_correct_local = 0
        mlm_total_local = 0
        num_batches_local = 0
        start_time = time.time()

        for input_ids, masked_tokens, masked_pos, cnn_features in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            masked_tokens = masked_tokens.to(device, non_blocking=True)
            masked_pos = masked_pos.to(device, non_blocking=True)
            cnn_features = cnn_features.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits_lm = model(input_ids, cnn_features, masked_pos)
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens).mean()
            correct, total = calculate_mlm_accuracy_counts(logits_lm, masked_tokens)
            loss_lm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_local += loss_lm.item()
            mlm_correct_local += correct
            mlm_total_local += total
            num_batches_local += 1
            global_step += 1

        # ---- DDP: aggregate training stats for logging ----
        train_loss_epoch = epoch_loss_local
        train_batches_epoch = float(num_batches_local)
        train_correct_epoch = float(mlm_correct_local)
        train_total_epoch = float(mlm_total_local)

        if ddp_is_initialized():
            train_loss_epoch = ddp_allreduce_scalar(train_loss_epoch, device)
            train_batches_epoch = ddp_allreduce_scalar(train_batches_epoch, device)
            train_correct_epoch = ddp_allreduce_scalar(train_correct_epoch, device)
            train_total_epoch = ddp_allreduce_scalar(train_total_epoch, device)

        epoch_loss = (train_loss_epoch / max(1.0, train_batches_epoch))
        mlm_acc_total = (train_correct_epoch / max(1.0, train_total_epoch))
        train_losses.append(epoch_loss)
        train_mlm_accuracies.append(mlm_acc_total)

        val_loss, val_accuracy = evaluate_bert_loader(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_mlm_accuracies.append(val_accuracy)

        if is_main_process():
            print(f"Train loss: {epoch_loss:.4f}, Train MLM Accuracy: {mlm_acc_total:.4f}")
            print(f"Val   loss: {val_loss:.4f}, Val   MLM Accuracy: {val_accuracy:.4f}")

        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # save only from rank 0
            if is_main_process():
                # unwrap model if DDP
                to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(to_save, bert_model_filename)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if is_main_process():
                    print(f"Early stopping after {epoch + 1} epochs. No improvement in validation loss for {patience} consecutive epochs.")
                break

        end_time = time.time()
        if is_main_process():
            print(f"{(end_time - start_time) / 60.0:.2f} minutes")

    if is_main_process():
        print("Model training is done !!!")
        pd.DataFrame({
            "epoch": list(range(1, len(train_losses)+1)),
            "train_loss": train_losses,
            "train_accuracy": train_mlm_accuracies,
            "val_loss": val_losses,
            "val_accuracy": val_mlm_accuracies,
        }).to_csv(train_results_filename, index=False)
        print(f"Train results have been saved to {train_results_filename}")

        fig, (ax_loss, ax_acc) = plt.subplots(nrows=2, figsize=(8, 8), sharex=False)
        ax_loss.plot(train_losses, label="train_loss")
        ax_loss.plot(val_losses, label="val_loss")
        ax_loss.set_ylabel("loss")
        ax_loss.legend()
        ax_acc.plot(train_mlm_accuracies, label="train_accuracy")
        ax_acc.plot(val_mlm_accuracies, label="val_accuracy")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")
        ax_acc.legend()
        fig.tight_layout()
        fig.savefig(train_plot_filename)
        plt.close(fig)
        print(f"Train plots have been saved to {train_plot_filename}")

    # optional: clean up DDP
    if ddp_is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
