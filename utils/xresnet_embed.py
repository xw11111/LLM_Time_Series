
# xresnet_embed.py  — clean, robust loader for fastai_xresnet1d101.pth
# - Loads legacy .pth (PyTorch <2.6 style) or a cleaned *_state_dict.pt
# - Extracts state_dict from many checkpoint formats (dict/module/nested)
# - Adapts first 1D conv from multi‑lead (e.g., 12) -> 1 lead by channel-mean
# - Exposes simple embedding APIs (segment-first or embed-once + ROI)

import os
import math
import sys
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from models.xresnet1d import xresnet1d101
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)



# ---------------------- small utilities ----------------------

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(mu): mu = 0.0
    if not np.isfinite(sd) or sd < eps: sd = 1.0
    return (x - mu) / (sd + eps)

def downsample_500_to_100(record_500hz: np.ndarray) -> np.ndarray:
    """Anti-aliased downsampling 500Hz -> 100Hz using polyphase filtering."""
    return resample_poly(record_500hz.astype(np.float32), up=1, down=5)

def scale_indices_500_to_100(start_500: int, end_500: int) -> Tuple[int, int]:
    s = int(round(start_500 / 5.0))
    e = int(round(end_500   / 5.0))
    if e <= s: e = s + 1
    return s, e

def pad_or_crop_center(sig: np.ndarray, target_len: int) -> np.ndarray:
    L = len(sig)
    if L == target_len: return sig
    if L < target_len:
        pad_total = target_len - L
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(sig, (pad_left, pad_right), mode="constant", constant_values=0.0)
    start = (L - target_len) // 2
    return sig[start:start+target_len]

# ---------------------- checkpoint helpers ----------------------

from collections.abc import Mapping
def _is_state_dict(obj) -> bool:
    return (
        isinstance(obj, Mapping) and obj and
        all(isinstance(k, str) for k in obj.keys()) and
        all(torch.is_tensor(v) or isinstance(v, torch.nn.Parameter) for v in obj.values())
    )

def _extract_state_dict(obj, _depth=0, _max_depth=5):
    """Find a state_dict in dict/module/list/tuple checkpoint objects."""
    if _is_state_dict(obj):
        return dict(obj)
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        try:
            sd = obj.state_dict()
            if _is_state_dict(sd): return dict(sd)
        except Exception:
            pass
    if isinstance(obj, Mapping):
        for key in ("state_dict","model_state","weights","params","model","net","module","ema","model_ema","teacher","student"):
            if key in obj:
                try:
                    return _extract_state_dict(obj[key], _depth+1, _max_depth)
                except Exception:
                    pass
        if _depth < _max_depth:
            for v in obj.values():
                try:
                    return _extract_state_dict(v, _depth+1, _max_depth)
                except Exception:
                    continue
    if isinstance(obj, (list, tuple)):
        for it in obj:
            try:
                return _extract_state_dict(it, _depth+1, _max_depth)
            except Exception:
                continue
    raise ValueError("Could not find a state_dict in the provided checkpoint.")

def _clean_state_dict_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        while True:
            if   k.startswith("module."): k = k[7:]
            elif k.startswith("model.") : k = k[6:]
            elif k.startswith("net.")   : k = k[4:]
            else: break
        out[k] = v
    return out

def _load_state_dict_from_any(path: str) -> dict:
    """
    Try three ways:
      1) torch.load(weights_only=True) with numpy-scalar allowlist
      2) torch.load(weights_only=False)  (only if you trust the file)
      3) torch.jit.load().state_dict()   (TorchScript)
    Return a cleaned flat state_dict (key -> Tensor).
    """
    # Prefer pre-cleaned *_state_dict.pt if available
    base, ext = os.path.splitext(path)
    state_dict_candidate = base + "_state_dict.pt"
    if os.path.exists(state_dict_candidate):
        obj = torch.load(state_dict_candidate, map_location="cpu", weights_only=True)
        sd = obj if _is_state_dict(obj) else _extract_state_dict(obj)
        return _clean_state_dict_keys(sd)

    # 1) Safe loader with allowlist for legacy numpy scalar
    try:
        from torch.serialization import add_safe_globals
        import numpy as _np
        try: add_safe_globals([_np.core.multiarray.scalar])
        except Exception: pass
        obj = torch.load(path, map_location="cpu", weights_only=True)
        sd  = _extract_state_dict(obj)
        return _clean_state_dict_keys(sd)
    except Exception:
        pass

    # 2) Legacy pickle path (trusted source only)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        sd  = _extract_state_dict(obj)
        return _clean_state_dict_keys(sd)
    except Exception:
        pass

    # 3) TorchScript archive
    try:
        script = torch.jit.load(path, map_location="cpu")
        sd = script.state_dict()
        return _clean_state_dict_keys(sd)
    except Exception as e:
        raise RuntimeError(f"Failed to obtain state_dict from checkpoint: {path}\n{e}")

# ---------------------- extractor ----------------------

class WaveEmbeddingExtractor:
    """
    Feature extractor for xresnet1d101 backbones.

    Usage:
        extractor = WaveEmbeddingExtractor("./models/fastai_xresnet1d101.pth", device="cuda", normalize=True)
        embs = extractor.get_embeddings_from_segments([seg1_500hz, seg2_500hz], batch_size=256)

    Methods:
      - get_embeddings_from_segments(segments_500hz, batch_size)
      - get_embeddings_from_full(record_500hz, seg_idx_500hz, batch_size)
      - get_feature_map_from_full(record_500hz)
      - roi_pool_segments(fmap, seg_idx_500hz, min_steps=4, margin_steps=2)
      - get_embeddings_roi_from_full(record_500hz, seg_idx_500hz, min_steps=4, margin_steps=2)
    """
    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 target_len: int = 256,
                 normalize: bool = True):
        self.device = torch.device(device if (device == "cpu") else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_len = int(target_len)
        self.normalize = bool(normalize)
        self.ds_factor = 5  # 500 Hz -> 100 Hz

        # 1) Build a 1‑channel xresnet and (later) drop its head for embeddings
        model_1ch = xresnet1d101(input_channels=1, num_classes=1)

        # 2) Load and clean state dict from checkpoint
        sd = _load_state_dict_from_any(model_path)
        

        # 3) Adapt first conv from multi‑lead -> single lead if needed
        first_conv_key = None
        for k in ("0.0.weight", "conv1.weight", "features.0.weight"):
            if k in sd: first_conv_key = k; break
        if first_conv_key is None:
            # fallback: pick the earliest [out_c, in_c, k] tensor
            three_d = [k for k, w in sd.items() if torch.is_tensor(w) and w.ndim == 3]
            if three_d:
                first_conv_key = sorted(three_d, key=lambda x: (x.count("."), len(x)))[0]

        if first_conv_key is not None:
            W = sd[first_conv_key]  # [C_out, C_in, k]
            if W.ndim == 3 and W.shape[1] != 1:
                sd[first_conv_key] = W.mean(dim=1, keepdim=True)  # channel-average 12->1, etc.

        # 4) Load weights (non-strict to ignore classifier/head mismatches)
        ret = model_1ch.load_state_dict(sd, strict=False)
        # Optional diagnostics:
        print(f"[load_state_dict] missing={ret.missing_keys[:8]} … unexpected={ret.unexpected_keys[:8]} …")

        # 5) Drop classifier head; keep the body for features
        self.body = nn.Sequential(*list(model_1ch.children())[:-1]).to(self.device)
        self.body.eval()

        # 6) Probe output channels and effective temporal stride at 100 Hz
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.target_len, device=self.device)
            feat = self.body(dummy)  # [1, C, L']
            self.out_channels = int(feat.shape[1])

            a, b = 2000, 3000  # samples @100 Hz
            f1 = self.body(torch.zeros(1, 1, a, device=self.device))
            f2 = self.body(torch.zeros(1, 1, b, device=self.device))
            d_in, d_out = (b - a), int(f2.shape[-1] - f1.shape[-1])
            self.eff_stride = int(round(d_in / d_out)) if d_out > 0 else max(1, a // max(1, int(f1.shape[-1])))

        self.embedding_dim = self.out_channels * 2  # avg + max
        print(f"Device: {self.device} | C={self.out_channels} | emb_dim={self.embedding_dim} | eff_stride@100Hz={self.eff_stride}")

    # ---------- segment-first APIs ----------

    def _prep_segment_500hz(self, seg_500hz: np.ndarray) -> np.ndarray:
        x100 = downsample_500_to_100(seg_500hz)
        if self.normalize: x100 = zscore(x100)
        x100 = pad_or_crop_center(x100, self.target_len)
        return x100.astype(np.float32)

    def _embed_batch(self, batch_100hz: np.ndarray, batch_size: int) -> np.ndarray:
        ds = torch.utils.data.TensorDataset(torch.from_numpy(batch_100hz))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
        outs = []
        with torch.no_grad():
            for (x,) in loader:
                x = x.unsqueeze(1).to(self.device)                    # [B,1,L]
                feat = self.body(x)                                   # [B,C,L']
                avg = F.adaptive_avg_pool1d(feat, 1)                  # [B,C,1]
                mx  = F.adaptive_max_pool1d(feat, 1)                  # [B,C,1]
                emb = torch.cat([avg, mx], dim=1).squeeze(-1)         # [B,2C]
                outs.append(emb.cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def get_embeddings_from_segments(self,
                                     segments_500hz: List[np.ndarray],
                                     batch_size: int = 256) -> np.ndarray:
        if not segments_500hz:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        batch = [self._prep_segment_500hz(seg) for seg in segments_500hz]
        return self._embed_batch(np.stack(batch), batch_size=batch_size)

    def get_embeddings_from_full(self,
                                 record_500hz: np.ndarray,
                                 seg_idx_500hz: List[Tuple[int, int]],
                                 batch_size: int = 256) -> np.ndarray:
        rec100 = downsample_500_to_100(record_500hz)
        if self.normalize: rec100 = zscore(rec100)

        batch = []
        for (s500, e500) in seg_idx_500hz:
            s100, e100 = scale_indices_500_to_100(s500, e500)
            s100 = max(0, min(s100, len(rec100)-1))
            e100 = max(0, min(e100, len(rec100)))
            if e100 <= s100: e100 = min(len(rec100), s100 + 1)
            seg = pad_or_crop_center(rec100[s100:e100], self.target_len)
            batch.append(seg.astype(np.float32))

        if not batch:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return self._embed_batch(np.stack(batch), batch_size=batch_size)

    # ---------- embed-once & ROI ----------

    def get_feature_map_from_full(self, record_500hz: np.ndarray) -> torch.Tensor:
        rec100 = downsample_500_to_100(record_500hz)
        if self.normalize: rec100 = zscore(rec100)
        x = torch.from_numpy(rec100.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,L]
        with torch.no_grad():
            fmap = self.body(x)  # [1,C,Lf]
        return fmap

    def roi_pool_segments(self,
                          fmap: torch.Tensor,
                          seg_idx_500hz: List[Tuple[int, int]],
                          min_steps: int = 4,
                          margin_steps: int = 2) -> np.ndarray:
        """
        ROI pool per segment over time-aligned feature map (embed-once workflow).
        Returns: [N_segments, 2*C] np.float32
        """
        assert fmap.ndim == 3 and fmap.shape[0] == 1, "fmap must be [1,C,Lf]"
        C, Lf = int(fmap.shape[1]), int(fmap.shape[2])
        embs = []
        for (s500, e500) in seg_idx_500hz:
            s100 = int(round(s500 / self.ds_factor))
            e100 = int(round(e500 / self.ds_factor))
            if e100 <= s100: e100 = s100 + 1
            sF = s100 // self.eff_stride
            eF = math.ceil(e100 / self.eff_stride)
            sF = max(0, sF - margin_steps)
            eF = min(Lf, eF + margin_steps)
            if (eF - sF) < min_steps:
                need = min_steps - (eF - sF)
                sF = max(0, sF - need // 2)
                eF = min(Lf, eF + (need - need // 2))
                if eF <= sF: eF = min(Lf, sF + 1)
            region = fmap[:, :, sF:eF]                   # [1,C,k]
            region = torch.nan_to_num(region, nan=0.0, posinf=0.0, neginf=0.0)
            avg = region.mean(dim=-1, keepdim=True)      # [1,C,1]
            mx  = region.amax(dim=-1, keepdim=True)      # [1,C,1]
            emb = torch.cat([avg, mx], dim=1).squeeze(0).squeeze(-1)  # [2C]
            embs.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.stack(embs, axis=0) if embs else np.zeros((0, 2*C), dtype=np.float32)

    def get_embeddings_roi_from_full(
        self,
        record_500hz: np.ndarray,
        seg_idx_500hz,               # accepts [(s,e), ...], or np.array Nx2, or dict with p/qrs/t
        min_steps: int = 4,
        margin_steps: int = 2,
        end_inclusive: bool = False, # set True if your end index is inclusive
    ) -> np.ndarray:
        n = int(len(record_500hz))            # 500 Hz * 10 s = 5000
        def _first_pair(x):
            if x is None: return None
            a = np.asarray(x)
            if a.ndim == 1 and a.size == 2:
                s, e = int(a[0]), int(a[1])
            elif a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 1:
                s, e = int(a[0,0]), int(a[0,1])    # take the FIRST pair if multiple
            else:
                return None
            if end_inclusive: e += 1               # convert [s,e] inclusive -> [s,e) exclusive
            s = max(0, min(s, n-1))                # clamp
            e = max(0, min(e, n))
            if e <= s: e = min(n, s+1)
            return (s, e)

        # Accept dict with p/qrs/t or a list/array directly
        items = seg_idx_500hz
        if isinstance(items, dict):
            items = [items.get(k) for k in ("p_idx","qrs_idx","t_idx")]
        pairs = [p for p in map(_first_pair, items) if p is not None]
        if not pairs:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        fmap = self.get_feature_map_from_full(record_500hz)   # [1,C,Lf]
        return self.roi_pool_segments(fmap, pairs, min_steps=min_steps, margin_steps=margin_steps)

