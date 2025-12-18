import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import boto3
import pickle
import pandas as pd
import neurokit2 as nk
import scipy
from tqdm import tqdm
import time
from random import shuffle, randint, randrange
import gc
import math
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torchsummary import summary
import warnings
warnings.simplefilter("ignore")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def init_distributed():
    if is_distributed():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def barrier():
    if is_distributed():
        dist.barrier()

def reduce_mean(tensor, device):
    if not is_distributed():
        return tensor
    t = tensor.clone().detach()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return t

def reduce_sum(tensor):
    if not is_distributed():
        return tensor
    t = tensor.clone().detach()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def save_weights(model, path):
    to_save = model.module if hasattr(model, "module") else model
    torch.save(to_save.state_dict(), path)

def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state)

set_seed(42)
init_distributed()
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
if get_rank() == 0:
    print(f"[DDP] world_size={get_world_size()}, device={DEVICE}")
torch.backends.cudnn.benchmark = True

config = {
    "use_batchnorm": True,
    "use_dropout": False,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "grid_len": 6,
    "epochs": 5,
    "dropout": 0.0,
    "patience": 10,
    "min_delta": 1e-5,
}

save_dir = "./pickles/new_pickles_800k_AE1/"
if get_rank() == 0:
    os.makedirs(save_dir, exist_ok=True)

use_avgmax = True
LAYERS   = 4
KERNELS  = [3, 3, 3, 3]
CHANNELS = [32, 64, 128, 256]
STRIDES  = [2, 2, 2, 2]

multi = 2 if use_avgmax else 1
LINEAR_DIM = multi * CHANNELS[-1] * config["grid_len"]

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        ts = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        return ts

class Encoder(nn.Module):
    def __init__(self, latent_dim, use_batchnorm=config["use_batchnorm"], use_dropout=config["use_dropout"]):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS
        self.strides = STRIDES
        self.conv = self.get_convs()
        self.linear = nn.Linear(LINEAR_DIM, latent_dim)

    def get_convs(self):
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            in_ch  = 1 if i == 0 else self.channels[i-1]
            out_ch = self.channels[i]
            conv_layers.append(nn.Conv1d(in_ch, out_ch,
                                         kernel_size=self.kernels[i],
                                         stride=self.strides[i],
                                         padding=self.kernels[i]//2))
            if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(nn.GELU())
            if self.use_dropout:
                conv_layers.append(nn.Dropout1d(config["dropout"]))
        return conv_layers

    def forward(self, x):
        x = self.conv(x)
        a = F.adaptive_avg_pool1d(x, config["grid_len"])
        if use_avgmax:
            m = F.adaptive_max_pool1d(x, config["grid_len"])
            zmap = torch.cat([a, m], dim=1)
        else:
            zmap = a
        z = self.linear(torch.flatten(zmap, 1))
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, use_batchnorm=config["use_batchnorm"], use_dropout=config["use_dropout"]):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS[::-1]
        self.strides = STRIDES
        self.linear = nn.Linear(latent_dim, CHANNELS[-1] * config["grid_len"])
        self.pre_up = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        self.conv = self.get_convs()
        self.output = nn.Conv1d(self.channels[-1], 1, kernel_size=1, stride=1)

    def get_convs(self):
        conv_layers = nn.Sequential()
        dec_chs = self.channels
        in_out_pairs = list(zip(dec_chs, dec_chs[1:] + [dec_chs[-1]]))
        for i, (in_ch, out_ch) in enumerate(in_out_pairs):
            conv_layers.append(nn.ConvTranspose1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=self.kernels[i],
                stride=self.strides[i],
                padding=self.kernels[i] // 2,
                output_padding=1
            ))
            if self.use_batchnorm and i != len(in_out_pairs) - 1:
                conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(nn.GELU())
            if self.use_dropout:
                conv_layers.append(nn.Dropout1d(config["dropout"]))
        return conv_layers

    def forward(self, z, target_length=None):
        C, L = CHANNELS[-1], config["grid_len"]
        x = self.linear(z).view(z.size(0), C, L)
        x = self.pre_up(x)
        x = self.conv(x)
        x = self.output(x)
        if target_length is not None and x.size(-1) != target_length:
            if x.size(-1) >= target_length:
                start = (x.size(-1) - target_length) // 2
                x = x[..., start:start+target_length]
            else:
                x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, use_batchnorm, use_dropout):
        super().__init__()
        self.encoder = Encoder(latent_dim, use_batchnorm, use_dropout)
        self.decoder = Decoder(latent_dim, use_batchnorm, use_dropout)

    def forward(self, x):
        target_length = x.size(-1)
        z = self.encoder(x)
        decoded = self.decoder(z, target_length=target_length)
        return decoded

def extract_waves_with_AE(wave_to_work, model_save_path, max_length, wave_type, latent_dim):

    padded_data = pad_sequences(wave_to_work, maxlen=max_length, padding='post', value=0.0, dtype='float32')
    wave_to_work = padded_data

    train_data, valid_data = train_test_split(wave_to_work, test_size=0.1, random_state=42)

    train_dataset = TimeSeriesDataset(train_data)
    valid_dataset = TimeSeriesDataset(valid_data)
    combined_dataset = TimeSeriesDataset(wave_to_work)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"],
        shuffle=False, sampler=valid_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    combined_loader = DataLoader(
        combined_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    model = AutoEncoder(latent_dim, config["use_batchnorm"], config["use_dropout"]).to(DEVICE)
    if get_rank() == 0:
        try:
            summary(model, (1, max_length))
        except Exception:
            pass

    if is_distributed():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False
        )

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    def train_one_epoch(model, dataloader):
        model.train()
        running = 0.0
        pbar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, disable=(get_rank()!=0), desc="Train")
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            x = batch.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                y_recons = model(x)
                loss = criterion(y_recons, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/max(1,len(dataloader)):.7f}")
            pbar.update()
            del x, y_recons
        pbar.close()
        return running / max(1, len(dataloader))

    def validate_epoch(model, dataloader):
        model.eval()
        running = 0.0
        pbar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, disable=(get_rank()!=0), desc="Valid")
        with torch.no_grad():
            for batch in dataloader:
                x = batch.to(DEVICE, non_blocking=True)
                y_recons = model(x)
                loss = criterion(y_recons, x)
                running += loss.item()
                pbar.set_postfix(loss=f"{running/max(1,len(dataloader)):.7f}")
                pbar.update()
                del x, y_recons
        pbar.close()
        local_val = torch.tensor([running / max(1, len(dataloader))], device=DEVICE)
        mean_val = reduce_mean(local_val, DEVICE)
        return float(mean_val.item())

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config["patience"]
    min_delta = config["min_delta"]
    best_model_path = model_save_path + ".tmp"

    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        if is_distributed():
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader)
        val_loss = validate_epoch(model, valid_loader)

        if get_rank() == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}  train={train_loss:.6f}  val={val_loss:.6f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if get_rank() == 0:
                save_weights(model, best_model_path)
                print(f"New best model saved to {best_model_path} !!!")
        else:
            epochs_no_improve += 1

        stop_flag = torch.tensor([1 if epochs_no_improve >= patience else 0], device=DEVICE)
        if is_distributed():
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            if get_rank() == 0:
                print(f"Early stopping: model doesn't improve for {patience} epochs")
            break

    barrier()
    load_weights(model, best_model_path, DEVICE)

    if get_rank() == 0:
        save_weights(model, model_save_path)
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        print(f"Best model ({best_val_loss:.6f}) copied to {model_save_path}")

        if len(train_losses) > 0:
            epochs_axis = range(1, len(train_losses) + 1)
            plt.figure(figsize=(5, 3))
            plt.plot(epochs_axis, train_losses, label="train_loss")
            plt.plot(epochs_axis, val_losses, label="val_loss")
            plt.xlabel("epoch"); plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"training_loss_{wave_type}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved training curve: training_loss_{wave_type}.png")
    def evaluate_metrics_ddp(model, dataloader, device=DEVICE, eps=1e-8):
        model.eval()
        net = model.module if hasattr(model, "module") else model

        n_local       = torch.tensor([0.0], device=device, dtype=torch.float64)
        s_y           = torch.tensor([0.0], device=device, dtype=torch.float64)
        s_yhat        = torch.tensor([0.0], device=device, dtype=torch.float64)
        s_y2          = torch.tensor([0.0], device=device, dtype=torch.float64)
        s_yhat2       = torch.tensor([0.0], device=device, dtype=torch.float64)
        s_yyhat       = torch.tensor([0.0], device=device, dtype=torch.float64)
        s_abs_err     = torch.tensor([0.0], device=device, dtype=torch.float64)
        s_sq_err      = torch.tensor([0.0], device=device, dtype=torch.float64)

        with torch.no_grad():
            for batch in dataloader:
                x = batch.to(device, non_blocking=True)
                yhat = net(x)

                mask = (x.abs() > eps).to(torch.bool)

                y     = x[mask].to(torch.float64)
                yhatm = yhat[mask].to(torch.float64)
                if y.numel() == 0:
                    continue

                n_local       += torch.tensor([y.numel()], device=device, dtype=torch.float64)
                s_y           += y.sum()
                s_yhat        += yhatm.sum()
                s_y2          += (y * y).sum()
                s_yhat2       += (yhatm * yhatm).sum()
                s_yyhat       += (y * yhatm).sum()
                s_abs_err     += (y - yhatm).abs().sum()
                s_sq_err      += ((y - yhatm) ** 2).sum()

                del x, yhat, y, yhatm, mask

        n        = reduce_sum(n_local)
        sy       = reduce_sum(s_y)
        syhat    = reduce_sum(s_yhat)
        sy2      = reduce_sum(s_y2)
        syhat2   = reduce_sum(s_yhat2)
        syyhat   = reduce_sum(s_yyhat)
        sae      = reduce_sum(s_abs_err)
        sse      = reduce_sum(s_sq_err)

        if get_rank() == 0:
            N = float(n.item())
            rmse = math.sqrt(float(sse.item()) / max(N, 1.0))
            mae  = float(sae.item()) / max(N, 1.0)

            num = N * float(syyhat.item()) - float(sy.item()) * float(syhat.item())
            den_left  = N * float(sy2.item())    - float(sy.item())**2
            den_right = N * float(syhat2.item()) - float(syhat.item())**2
            den = math.sqrt(max(den_left, 0.0) * max(den_right, 0.0))
            r = num / den if den > 0 else float('nan')

            sst = float(sy2.item()) - (float(sy.item())**2) / max(N, 1.0)
            r2  = 1.0 - (float(sse.item()) / sst) if sst > 0 else float('nan')

            print("\n[Best-model Metrics on validation (masked, no padding)]")
            print(f"  Pearson r: {r:.6f}")
            print(f"  RMSE     : {rmse:.6f}")
            print(f"  MAE      : {mae:.6f}")
            print(f"  R²       : {r2:.6f}")

    evaluate_metrics_ddp(model, valid_loader)

    barrier()

    if get_rank() == 0:
        def extract_latent_space(data_loader, model, device=DEVICE):
            latents = []
            net = model.module if hasattr(model, "module") else model
            enc = net.encoder
            net.eval()
            with torch.no_grad():
                for batch in data_loader:
                    x = batch.to(device, non_blocking=True)
                    z = enc(x)
                    latents.append(z.cpu())
            return torch.cat(latents, dim=0).numpy()

        latent_space_wave = extract_latent_space(combined_loader, model)
        print(f"Latent space shape: {latent_space_wave.shape}")
        print(f"Combined data shape: {wave_to_work.shape}")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{wave_type}_latent_space.npy")
        np.save(save_path, latent_space_wave.astype(np.float32))
        print(f"Latent vectors saved to {save_path}  (shape: {latent_space_wave.shape})")

    barrier()

def encode_waves(waves, ae_weight, max_len, z_dim, batch_size, device=DEVICE, grid_len=7):

    state = torch.load(ae_weight, map_location="cpu")
    lin_w = state.get("encoder.linear.weight", None)
    if lin_w is None:
        lin_w = state.get("module.encoder.linear.weight", None)
    if lin_w is None:

        for k, v in state.items():
            if k.endswith("encoder.linear.weight"):
                lin_w = v
                break
    if lin_w is None:
        raise RuntimeError("Missing encoder.linear.weight in checkpoint; cannot infer/validate grid_len.")
    if lin_w is None:
        raise RuntimeError("Missing encoder.linear.weight in checkpoint; cannot infer grid_len.")

    C = CHANNELS[-1]

    global LINEAR_DIM, config, use_avgmax
    use_avgmax = True
    config["grid_len"] = int(grid_len)
    LINEAR_DIM = 2 * CHANNELS[-1] * config["grid_len"]

    expected_F = 2 * C * config["grid_len"]
    if lin_w.shape[1] != expected_F:
        raise ValueError(
            f"grid_len mismatch: checkpoint expects flattened dim {lin_w.shape[1]}, "
            f"but with grid_len={config['grid_len']} (use_avgmax=True) the model expects {expected_F}. "
            f"Pass grid_len={lin_w.shape[1] // (2*C)} to match the checkpoint."
        )

    waves = pad_sequences(waves, maxlen=max_len, padding='post', value=0.0, dtype='float32')
    loader = DataLoader(TimeSeriesDataset(waves), batch_size=batch_size, shuffle=False, pin_memory=True)

    ae = AutoEncoder(z_dim, use_batchnorm=config['use_batchnorm'], use_dropout=config['use_dropout']).to(device)
    ae.load_state_dict(state)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    latents = []
    with torch.no_grad():
        for x in loader:
            z = ae.encoder(x.to(device, non_blocking=True))
            latents.append(z.cpu())
    return torch.cat(latents).numpy()

__all__ = [
    "extract_waves_with_AE",
    "encode_waves"
]

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--wave_type", required=True, choices=["p_wave","qrs_wave","t_wave"])
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--max_len", type=int, required=True)

    bn_grp = parser.add_mutually_exclusive_group()
    bn_grp.add_argument("--use_batchnorm",  dest="use_batchnorm",  action="store_true")
    bn_grp.add_argument("--no_batchnorm",   dest="use_batchnorm",  action="store_false")
    parser.set_defaults(use_batchnorm=config["use_batchnorm"])

    do_grp = parser.add_mutually_exclusive_group()
    do_grp.add_argument("--use_dropout",    dest="use_dropout",    action="store_true")
    do_grp.add_argument("--no_dropout",     dest="use_dropout",    action="store_false")
    parser.set_defaults(use_dropout=config["use_dropout"])

    parser.add_argument("--lr",            type=float, default=config["lr"])
    parser.add_argument("--weight_decay",  type=float, default=config["weight_decay"])
    parser.add_argument("--batch_size",    type=int,   default=config["batch_size"])
    parser.add_argument("--grid_len",      type=int,   default=config["grid_len"])
    parser.add_argument("--epochs",        type=int,   default=config["epochs"])
    parser.add_argument("--dropout",       type=float, default=config["dropout"])
    parser.add_argument("--patience",      type=int,   default=config["patience"])
    parser.add_argument("--min_delta",     type=float, default=config["min_delta"])

    args = parser.parse_args()

    override_keys = ["use_batchnorm", "use_dropout", "lr", "weight_decay",
                     "batch_size", "grid_len", "epochs", "dropout", "patience", "min_delta"]
    for k in override_keys:
        config[k] = getattr(args, k)

    multi = 2 if use_avgmax else 1
    LINEAR_DIM = multi * CHANNELS[-1] * config["grid_len"]

    arr = np.load(args.data, allow_pickle=True)
    extract_waves_with_AE(
        wave_to_work=arr,
        model_save_path=args.model_out,
        max_length=args.max_len,
        wave_type=args.wave_type,
        latent_dim=args.latent_dim
    )