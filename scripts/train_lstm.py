import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from src.LSTMonly import LSTMOnlyModel
from data.dataloader import KeypointDataset, WindowedKeypointDataset

from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm

import wandb
wandb.login(key="8b49b325ce8e9e788b2981b63eebbc01ee33bc6b")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- fake dataset example -----
SAVE_INTERVAL = 1
NUM_EPOCHS = 150
BATCH = 64
O = 50
P = 10
STRIDE = 10
Normalize = False
Smoothing = False
Smoothing_window = 5
M = 2               # Number of instruments
lr = 1e-4
Enable_WandB = True
hidden_size = 128
num_layers=2
Transformer_used = False

w_pos = 1.0
w_delta = 0.3
w_dir = 0.2
w_mag = 0.3

base = Path("/raid/home/patrickbyc/SurgPose_dataset_no_vid")
paths_left = list(base.rglob("keypoints_left.yaml"))
# paths_right = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_right.yaml", recursive=True)

yaml_paths = paths_left

print("num yamls:", len(yaml_paths))

if len(yaml_paths) == 0:
    raise RuntimeError("No YAML files found. Check the path and recursive glob.")

ds = KeypointDataset(
    yaml_paths=yaml_paths,
    normalize=Normalize,
    smoothing=Smoothing_window,
    smoothing_window=5,
)

# 80 / 20 split
n_total = len(ds)
n_test = int(0.2 * n_total)
n_train = n_total - n_test

train_base, test_base = random_split(
    ds,
    [n_train, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_set = WindowedKeypointDataset(train_base, O=O, P=P, random_window=False, load_from_image=False, stride=STRIDE)
test_set  = WindowedKeypointDataset(test_base,  O=O, P=P, random_window=False, load_from_image=False, stride=STRIDE)
print("num train_set:", len(train_set))
print("num test_set:", len(test_set))
train_dl = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_dl  = DataLoader(test_set,  batch_size=BATCH, shuffle=False)


wandb_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
if Enable_WandB:
    job_name = "EECE571F_Project_training" + "dataset: surgpose" + "LSTM Benchmark"
    args = {
        "lr": lr,
        "hidden_size": 128,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH,
        "Obs_frames": O,
        "Pred_frames": P,
        "stride": STRIDE,
        "normalize": Normalize,
        "smoothing": Smoothing,
        "smoothing_window": Smoothing_window,
        "Transformer_used": Transformer_used,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "w_pos": w_pos,
        "w_delta": w_delta,
        "w_dir": w_dir,
        "w_mag": w_mag,
        "num_keypoint_yaml": len(yaml_paths),
    }
    wandb.init(
        project="EECE571F_Project",
        name=job_name,
    )
    wandb.config.current_time = wandb_timestamp
    wandb.config.update(args)


# ----- build model -----
model = LSTMOnlyModel(
    M=M,
    hidden_size=hidden_size,
    num_layers=num_layers,
).to(device)

optimizer = Adam(model.parameters(), lr=lr)

def _as_delta_pred(model_out):
    """
    model_out can be:
      (B,T,N,2) or (B,T,N,4) or (B,T,1,N,2/4)
    return mu: (B,T,N,2)
    """
    if model_out.ndim == 5:
        model_out = model_out.squeeze(2)  # (B,T,N,C)
    C = model_out.shape[-1]
    if C == 2:
        return model_out
    elif C == 4:
        return model_out[..., 0:2]  # mu only
    else:
        raise RuntimeError(f"Unexpected model output last dim C={C}")

def add_virtual_root_from_xy(feat):
    """
    feat: (B,T,M,5,3) where last dim = [x, y, valid]
    returns: (B,T,M,6,3)
    root = mean of valid keypoints
    """
    xy = feat[..., :2]                    # (B,T,M,5,2)
    valid = feat[..., 2] > 0.5            # (B,T,M,5)

    valid_f = valid.unsqueeze(-1).float() # (B,T,M,5,1)
    denom = valid_f.sum(dim=3, keepdim=True).clamp_min(1.0)
    root_xy = (xy * valid_f).sum(dim=3, keepdim=True) / denom   # (B,T,M,1,2)
    root_valid = valid.any(dim=3, keepdim=True).float().unsqueeze(-1)  # (B,T,M,1,1)

    root = torch.cat([root_xy, root_valid], dim=-1)             # (B,T,M,1,3)
    return torch.cat([feat, root], dim=3)                       # (B,T,M,6,3)

def _make_gt_and_mask(model, pos_raw):
    """
    pos_raw: (B,T,M,5,2)

    returns:
      gt_delta:  (B,T-1,N,2)
      mask:      (B,T-1,N,2)
      delta_in:  (B,T-1,M,5,2)
      pos_clean: (B,T,M,5,2)
    """
    valid_xy = torch.isfinite(pos_raw).all(dim=-1)          # (B,T,M,5)
    pos_clean = torch.nan_to_num(pos_raw, nan=0.0)          # (B,T,M,5,2)

    feat = torch.cat([
        pos_clean,
        valid_xy.unsqueeze(-1).float()
    ], dim=-1)                                              # (B,T,M,5,3)

    pos6_feat = add_virtual_root_from_xy(feat)        # (B,T,M,6,3)
    pos6 = pos6_feat[..., :2]                               # (B,T,M,6,2)

    gt_delta = pos6[:, 1:] - pos6[:, :-1]                   # (B,T-1,M,6,2)
    gt_delta = gt_delta.reshape(gt_delta.size(0), gt_delta.size(1), -1, 2)

    delta_xy = pos_clean[:, 1:] - pos_clean[:, :-1]         # (B,T-1,M,5,2)
    valid_delta5 = valid_xy[:, 1:] & valid_xy[:, :-1]       # (B,T-1,M,5)
    delta_xy = torch.where(valid_delta5.unsqueeze(-1), delta_xy, torch.zeros_like(delta_xy))

    delta_in = torch.cat([
        delta_xy,
        valid_delta5.unsqueeze(-1).float()
    ], dim=-1)            

    root_valid_t = valid_xy.any(dim=-1)                     # (B,T,M)
    root_valid_dt = root_valid_t[:, 1:] & root_valid_t[:, :-1]   # (B,T-1,M)

    valid_delta6 = torch.cat([valid_delta5, root_valid_dt.unsqueeze(-1)], dim=-1)  # (B,T-1,M,6)
    mask = valid_delta6.unsqueeze(-1).expand(-1, -1, -1, -1, 2).float()
    mask = mask.reshape(mask.size(0), mask.size(1), -1, 2)

    return gt_delta, mask, delta_in, pos_clean

@torch.no_grad()
def _direction_loss(pred_delta, gt_delta, mask=None, eps=1e-8, min_speed=1.0):
    """
    pred_delta, gt_delta: (..., 2)
    mask: (...,) boolean or float
    """
    gt_norm = torch.norm(gt_delta, dim=-1)          # (...)
    pred_norm = torch.norm(pred_delta, dim=-1)      # (...)

    valid = gt_norm > min_speed
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask > 0
        valid = valid & mask

    if not valid.any():
        return pred_delta.new_tensor(0.0)

    pred_unit = pred_delta / (pred_norm.unsqueeze(-1) + eps)
    gt_unit   = gt_delta   / (gt_norm.unsqueeze(-1) + eps)

    cos = (pred_unit * gt_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos[valid]).mean()


def train_one_epoch(
    model, dataloader, optimizer, device, O, P,
    w_pos=1.0, w_delta=0.5, w_dir=0.2, w_mag=0.1, grad_clip=1.0
):
    model.train()
    total, n = 0.0, 0

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for batch in pbar:
        obs_raw = batch["obs"].to(device).float()
        fut_raw = batch["fut"].to(device).float()

        obs_frames  = batch["obs_frames"].to(device).float()
        full_frames = batch["full_frames"].to(device).float()

        obs = torch.nan_to_num(obs_raw, nan=0.0)
        fut = torch.nan_to_num(fut_raw, nan=0.0)

        # -------------------------
        # (A) Teacher-forced delta loss
        # -------------------------
        full_raw = torch.cat([obs_raw, fut_raw], dim=1)
        gt_delta, mask_delta, full_delta_in, full_clean = _make_gt_and_mask(model, full_raw)

        out_tf = model(full_delta_in, full_frames)
        pred_mu_tf = _as_delta_pred(out_tf)   # (B,O+P-1,N,2) expected
        pred_delta_tf = pred_mu_tf

        delta_loss = ((pred_delta_tf - gt_delta) ** 2 * mask_delta).sum() / mask_delta.sum().clamp_min(1.0)

        # -------------------------
        # (B) Rollout absolute position loss
        # -------------------------
        fut_valid5 = torch.isfinite(fut_raw).all(dim=-1)
        fut_clean = torch.nan_to_num(fut_raw, nan=0.0)
        fut_feat = torch.cat([
            fut_clean,
            fut_valid5.unsqueeze(-1).float()
        ], dim=-1)   # (B,P,M,5,3)

        gt_fut6 = add_virtual_root_from_xy(fut_feat)[..., :2]   # (B,P,M,6,2)

        root_valid = fut_valid5.any(dim=-1, keepdim=True)
        valid6 = torch.cat([fut_valid5, root_valid], dim=-1)          # (B,P,M,6)
        mask6 = valid6.unsqueeze(-1).float()                          # (B,P,M,6,1)

        B, _, M, _, _ = obs.shape

        seq_raw = obs_raw.clone()
        seq_clean = obs.clone()
        seq_valid5 = torch.isfinite(seq_raw).all(dim=-1)

        seq_delta_xy = seq_clean[:, 1:] - seq_clean[:, :-1]
        seq_delta_valid = seq_valid5[:, 1:] & seq_valid5[:, :-1]
        seq_delta_xy = torch.where(seq_delta_valid.unsqueeze(-1), seq_delta_xy, torch.zeros_like(seq_delta_xy))

        seq_delta = torch.cat([
            seq_delta_xy,
            seq_delta_valid.unsqueeze(-1).float()
        ], dim=-1)   # (B,O-1,M,5,3)

        seq_frames = obs_frames.clone()

        preds6 = []
        pred_rollout_deltas6 = []

        for _step in range(P):
            out = model(seq_delta, seq_frames)
            mu = _as_delta_pred(out)
            dposN = mu[:, -1]                       # (B,N,2)
            dpos6 = dposN.view(B, M, 6, 2)         # (B,M,6,2)

            last5_raw = seq_raw[:, -1]
            last_valid5 = seq_valid5[:, -1]
            last5_clean = torch.nan_to_num(last5_raw, nan=0.0)
            last_feat = torch.cat([
                last5_clean.unsqueeze(1),
                last_valid5.unsqueeze(1).unsqueeze(-1).float()
            ], dim=-1)

            last6 = add_virtual_root_from_xy(last_feat)[:, 0, ..., :2]   # (B,M,6,2)

            next6 = last6 + dpos6
            next5 = next6[:, :, :5, :]
            next_valid5 = torch.ones((B, M, 5), device=device, dtype=torch.bool)

            next_feat = torch.cat([
                next5.unsqueeze(1),
                next_valid5.unsqueeze(1).unsqueeze(-1).float()
            ], dim=-1)

            next6 = add_virtual_root_from_xy(next_feat)[:, 0, ..., :2]

            preds6.append(next6)
            pred_rollout_deltas6.append(next6 - last6)   # actual rollout delta used

            seq_raw = torch.cat([seq_raw[:, 1:], next5.unsqueeze(1)], dim=1)
            seq_valid5 = torch.cat([seq_valid5[:, 1:], next_valid5.unsqueeze(1)], dim=1)
            seq_clean = torch.nan_to_num(seq_raw, nan=0.0)

            seq_delta_xy = seq_clean[:, 1:] - seq_clean[:, :-1]
            seq_delta_valid = seq_valid5[:, 1:] & seq_valid5[:, :-1]
            seq_delta_xy = torch.where(seq_delta_valid.unsqueeze(-1), seq_delta_xy, torch.zeros_like(seq_delta_xy))

            seq_delta = torch.cat([
                seq_delta_xy,
                seq_delta_valid.unsqueeze(-1).float()
            ], dim=-1)

            last_frame = seq_frames[:, -1:].clone()
            seq_frames = torch.cat([seq_frames[:, 1:], last_frame], dim=1)

        pred_fut6 = torch.stack(preds6, dim=1)                    # (B,P,M,6,2)
        pred_rollout_deltas6 = torch.stack(pred_rollout_deltas6, dim=1)  # (B,P,M,6,2)

        pos_loss = (((pred_fut6 - gt_fut6) ** 2) * mask6).sum() / mask6.sum().clamp_min(1.0)

        # -------------------------
        # (C) Direction loss on rollout deltas
        # -------------------------
        gt_prev6 = torch.cat([gt_fut6[:, :1], gt_fut6[:, :-1]], dim=1)   # temporary
        # first future step should be relative to last observed frame, not gt_fut6[:, :1]
        last_obs5_raw = obs_raw[:, -1]
        last_obs5_valid = torch.isfinite(last_obs5_raw).all(dim=-1)
        last_obs5_clean = torch.nan_to_num(last_obs5_raw, nan=0.0)
        last_obs_feat = torch.cat([
            last_obs5_clean.unsqueeze(1),
            last_obs5_valid.unsqueeze(1).unsqueeze(-1).float()
        ], dim=-1)
        last_obs6 = add_virtual_root_from_xy(last_obs_feat)[..., :2]   # (B,1,M,6,2)

        gt_prev6 = torch.cat([last_obs6, gt_fut6[:, :-1]], dim=1)            # (B,P,M,6,2)
        gt_rollout_deltas6 = gt_fut6 - gt_prev6                              # (B,P,M,6,2)

        dir_mask = valid6   # (B,P,M,6)
        dir_loss = _direction_loss(
            pred_rollout_deltas6,
            gt_rollout_deltas6,
            mask=dir_mask,
            min_speed=1.0
        )

        # optional magnitude loss, useful if predictions are too short
        mag_loss = (
            (
                torch.norm(pred_rollout_deltas6, dim=-1) -
                torch.norm(gt_rollout_deltas6, dim=-1)
            ) ** 2
        )
        mag_loss = (mag_loss * dir_mask.float()).sum() / dir_mask.float().sum().clamp_min(1.0)

        # -------------------------
        # Final loss
        # -------------------------
        loss = w_pos * pos_loss + w_delta * delta_loss + w_dir * dir_loss + w_mag * mag_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += float(loss.item())
        n += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pos": f"{pos_loss.item():.4f}",
            "delta": f"{delta_loss.item():.4f}",
            "dir": f"{dir_loss.item():.4f}",
            "mag": f"{mag_loss.item():.4f}",
        })

    return total / max(n, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model, dataloader, device, O, P,
    w_pos=1.0, w_delta=0.5, w_dir=0.2, w_mag=0.1
):
    model.eval()
    total, n = 0.0, 0
    total_pos, total_delta = 0.0, 0.0
    total_dir, total_mag = 0.0, 0.0

    pbar = tqdm(dataloader, desc="Eval", leave=False)

    for batch in pbar:
        obs_raw = batch["obs"].to(device).float()
        fut_raw = batch["fut"].to(device).float()
        obs_frames  = batch["obs_frames"].to(device).float()
        full_frames = batch["full_frames"].to(device).float()

        obs = torch.nan_to_num(obs_raw, nan=0.0)
        fut = torch.nan_to_num(fut_raw, nan=0.0)

        # --------------------------------
        # (A) Teacher-forced delta loss
        # --------------------------------
        full_raw = torch.cat([obs_raw, fut_raw], dim=1)
        gt_delta, mask_delta, full_delta_in, full_clean = _make_gt_and_mask(model, full_raw)

        out_tf = model(full_delta_in, full_frames)
        pred_mu_tf = _as_delta_pred(out_tf)
        pred_delta_tf = pred_mu_tf

        delta_loss = ((pred_delta_tf - gt_delta) ** 2 * mask_delta).sum() / mask_delta.sum().clamp_min(1.0)

        # --------------------------------
        # (B) GT future absolute positions
        # --------------------------------
        fut_valid5 = torch.isfinite(fut_raw).all(dim=-1)
        fut_clean = torch.nan_to_num(fut_raw, nan=0.0)
        fut_feat = torch.cat([
            fut_clean,
            fut_valid5.unsqueeze(-1).float()
        ], dim=-1)   # (B,P,M,5,3)

        gt_fut6 = add_virtual_root_from_xy(fut_feat)[..., :2]   # (B,P,M,6,2)

        root_valid = fut_valid5.any(dim=-1, keepdim=True)
        valid6 = torch.cat([fut_valid5, root_valid], dim=-1)          # (B,P,M,6)
        mask6 = valid6.unsqueeze(-1).float()                          # (B,P,M,6,1)

        B, _, M, _, _ = obs.shape

        # --------------------------------
        # (C) Autoregressive rollout
        # --------------------------------
        seq_raw = obs_raw.clone()
        seq_clean = obs.clone()
        seq_valid5 = torch.isfinite(seq_raw).all(dim=-1)

        seq_delta_xy = seq_clean[:, 1:] - seq_clean[:, :-1]
        seq_delta_valid = seq_valid5[:, 1:] & seq_valid5[:, :-1]
        seq_delta_xy = torch.where(seq_delta_valid.unsqueeze(-1), seq_delta_xy, torch.zeros_like(seq_delta_xy))

        seq_delta = torch.cat([
            seq_delta_xy,
            seq_delta_valid.unsqueeze(-1).float()
        ], dim=-1)   # (B,O-1,M,5,3)

        seq_frames = obs_frames.clone()

        preds6 = []
        pred_rollout_deltas6 = []

        for _step in range(P):
            out = model(seq_delta, seq_frames)
            mu = _as_delta_pred(out)
            dposN = mu[:, -1]
            dpos6 = dposN.view(B, M, 6, 2)

            last5_raw = seq_raw[:, -1]
            last_valid5 = seq_valid5[:, -1]
            last5_clean = torch.nan_to_num(last5_raw, nan=0.0)
            last_feat = torch.cat([
                last5_clean.unsqueeze(1),
                last_valid5.unsqueeze(1).unsqueeze(-1).float()
            ], dim=-1)

            last6 = add_virtual_root_from_xy(last_feat)[:, 0, ..., :2]   # (B,M,6,2)

            next6 = last6 + dpos6
            next5 = next6[:, :, :5, :]
            next_valid5 = torch.ones((B, M, 5), device=device, dtype=torch.bool)

            next_feat = torch.cat([
                next5.unsqueeze(1),
                next_valid5.unsqueeze(1).unsqueeze(-1).float()
            ], dim=-1)

            next6 = add_virtual_root_from_xy(next_feat)[:, 0, ..., :2]

            preds6.append(next6)
            pred_rollout_deltas6.append(next6 - last6)

            seq_raw = torch.cat([seq_raw[:, 1:], next5.unsqueeze(1)], dim=1)
            seq_valid5 = torch.cat([seq_valid5[:, 1:], next_valid5.unsqueeze(1)], dim=1)
            seq_clean = torch.nan_to_num(seq_raw, nan=0.0)

            seq_delta_xy = seq_clean[:, 1:] - seq_clean[:, :-1]
            seq_delta_valid = seq_valid5[:, 1:] & seq_valid5[:, :-1]
            seq_delta_xy = torch.where(seq_delta_valid.unsqueeze(-1), seq_delta_xy, torch.zeros_like(seq_delta_xy))

            seq_delta = torch.cat([
                seq_delta_xy,
                seq_delta_valid.unsqueeze(-1).float()
            ], dim=-1)

            last_frame = seq_frames[:, -1:].clone()
            seq_frames = torch.cat([seq_frames[:, 1:], last_frame], dim=1)

        pred_fut6 = torch.stack(preds6, dim=1)                          # (B,P,M,6,2)
        pred_rollout_deltas6 = torch.stack(pred_rollout_deltas6, dim=1) # (B,P,M,6,2)

        pos_loss = (((pred_fut6 - gt_fut6) ** 2) * mask6).sum() / mask6.sum().clamp_min(1.0)

        # --------------------------------
        # (D) Direction + magnitude loss
        # --------------------------------
        last_obs5_raw = obs_raw[:, -1]
        last_obs5_valid = torch.isfinite(last_obs5_raw).all(dim=-1)
        last_obs5_clean = torch.nan_to_num(last_obs5_raw, nan=0.0)
        last_obs_feat = torch.cat([
            last_obs5_clean.unsqueeze(1),
            last_obs5_valid.unsqueeze(1).unsqueeze(-1).float()
        ], dim=-1)

        last_obs6 = add_virtual_root_from_xy(last_obs_feat)[..., :2]  # (B,1,M,6,2)

        gt_prev6 = torch.cat([last_obs6, gt_fut6[:, :-1]], dim=1)            # (B,P,M,6,2)
        gt_rollout_deltas6 = gt_fut6 - gt_prev6                              # (B,P,M,6,2)

        dir_mask = valid6
        dir_loss = _direction_loss(
            pred_rollout_deltas6,
            gt_rollout_deltas6,
            mask=dir_mask,
            min_speed=1.0
        )

        mag_loss = (
            (
                torch.norm(pred_rollout_deltas6, dim=-1) -
                torch.norm(gt_rollout_deltas6, dim=-1)
            ) ** 2
        )
        mag_loss = (mag_loss * dir_mask.float()).sum() / dir_mask.float().sum().clamp_min(1.0)

        combined = w_pos * pos_loss + w_delta * delta_loss + w_dir * dir_loss + w_mag * mag_loss

        if torch.isnan(combined) or torch.isinf(combined):
            continue

        total += float(combined.item())
        total_pos += float(pos_loss.item())
        total_delta += float(delta_loss.item())
        total_dir += float(dir_loss.item())
        total_mag += float(mag_loss.item())
        n += 1

    if n == 0:
        return {
            "loss": float("nan"),
            "pos_loss": float("nan"),
            "delta_loss": float("nan"),
            "dir_loss": float("nan"),
            "mag_loss": float("nan"),
        }

    return {
        "loss": total / n,
        "pos_loss": total_pos / n,
        "delta_loss": total_delta / n,
        "dir_loss": total_dir / n,
        "mag_loss": total_mag / n,
    }

# ----- training -----
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_dl, optimizer, device, O, P, w_pos=w_pos, w_delta=w_delta, w_dir=w_dir, w_mag=w_mag)
    metrics = evaluate_one_epoch(model, test_dl, device, O, P, w_pos=w_pos, w_delta=w_delta, w_dir=w_dir, w_mag=w_mag)
    
    print(
        f"epoch {epoch}: "
        f"train={train_loss:.6f}, "
        f"test_total={metrics['loss']:.6f}, "
        f"test_pos={metrics['pos_loss']:.6f}, "
        f"test_delta={metrics['delta_loss']:.6f}, "
        f"test_dir={metrics['dir_loss']:.6f}, "
        f"test_mag={metrics['mag_loss']:.6f}"
    )
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": metrics["loss"],
        "test_pos": metrics["pos_loss"],
        "test_delta": metrics["delta_loss"],
        "test_dir": metrics["dir_loss"],
        "test_mag": metrics["mag_loss"],
    })

    if (epoch + 1) % SAVE_INTERVAL == 0:
        os.makedirs(f"models_lstm/model_weights_{wandb_timestamp}", exist_ok=True)
        torch.save(model.state_dict(), f"models_lstm/model_weights_{wandb_timestamp}/epoch{epoch}.pth")