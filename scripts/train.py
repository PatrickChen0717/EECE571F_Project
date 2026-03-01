import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from src.LSTM import LSTM_gat
from src.GAT import GAT
from src.model import FullModel
from data.dataloader import KeypointDataset, WindowedKeypointDataset

import glob
import numpy as np

import wandb
wandb.login(key="8b49b325ce8e9e788b2981b63eebbc01ee33bc6b")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- fake dataset example -----
SAVE_INTERVAL = 3
NUM_EPOCHS = 150
BATCH = 32
O = 30
P = 30
M = 2               # Number of instruments
lr = 1e-3
Enable_WandB = True

paths_left = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_left.yaml", recursive=True)
# paths_right = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_right.yaml", recursive=True)

yaml_paths = paths_left

print("num yamls:", len(yaml_paths))

if len(yaml_paths) == 0:
    raise RuntimeError("No YAML files found. Check the path and recursive glob.")

ds = KeypointDataset(
    yaml_paths=yaml_paths,
    normalize=False
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

train_set = WindowedKeypointDataset(train_base, O=O, P=P, random_window=True)
test_set  = WindowedKeypointDataset(test_base,  O=O, P=P, random_window=False)
print("num train_set:", len(train_set))
print("num test_set:", len(test_set))
train_dl = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_dl  = DataLoader(test_set,  batch_size=BATCH, shuffle=False)


wandb_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
if Enable_WandB:
    job_name = "EECE571F_Project_training" + "dataset: surgpose" + str(len(yaml_paths))
    args = {
        "lr": lr,
        "hidden_size": 128,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH,
        "Obs_frames": O,
        "Pred_frames": P,
        "num_keypoint_yaml": len(yaml_paths),
    }
    wandb.init(
        project="EECE571F_Project",
        name=job_name,
    )
    wandb.config.current_time = wandb_timestamp
    wandb.config.update(args)


# ----- build model -----
encoder = LSTM_gat(hidden_size=128, embed_dim=64)
model = FullModel(encoder, gat_out_dim=128).to(device)

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


def _make_gt_and_mask(model, pos_raw):
    """
    pos_raw: (B,T,M,5,2) with NaNs possible
    returns:
      gt_delta: (B,T-1,N,2) where N=M*6
      mask:     (B,T-1,N,2) float
      pos_clean:(B,T,M,5,2) cleaned (NaNs->0) for model input
    """
    # validity per kp (True if both x,y finite)
    valid_xy = torch.isfinite(pos_raw).all(dim=-1)          # (B,T,M,5)
    pos_clean = torch.nan_to_num(pos_raw, nan=0.0)          # (B,T,M,5,2)

    # IMPORTANT: build pos6 from RAW + valid mask, not from cleaned tensor
    pos6 = model.encoder.add_virtual_root(pos_raw, valid_xy)  # (B,T,M,6,2) root LAST

    gt_delta = pos6[:, 1:] - pos6[:, :-1]                   # (B,T-1,M,6,2)
    gt_delta = gt_delta.reshape(gt_delta.size(0), gt_delta.size(1), -1, 2)  # (B,T-1,N,2)

    # mask for delta valid: endpoints must be valid
    valid_delta5 = valid_xy[:, 1:] & valid_xy[:, :-1]       # (B,T-1,M,5)

    root_valid_t  = valid_xy.any(dim=-1)                    # (B,T,M) root valid if any kp valid
    root_valid_dt = root_valid_t[:, 1:] & root_valid_t[:, :-1]  # (B,T-1,M)

    # node order matches add_virtual_root: [kp1..kp5, root]
    valid_delta6 = torch.cat([valid_delta5, root_valid_dt.unsqueeze(-1)], dim=-1)  # (B,T-1,M,6)
    mask = valid_delta6.unsqueeze(-1).expand(-1, -1, -1, -1, 2).float()
    mask = mask.reshape(mask.size(0), mask.size(1), -1, 2)

    return gt_delta, mask, pos_clean


def train_one_epoch(
    model, dataloader, optimizer, device, O, P,
    w_pos=1.0, w_delta=0.5, grad_clip=1.0
):
    model.train()
    total, n = 0.0, 0

    for batch in dataloader:
        obs_raw = batch["obs"].to(device).float()  # (B,O,M,5,2) with NaNs possible
        fut_raw = batch["fut"].to(device).float()  # (B,P,M,5,2) with NaNs possible

        # cleaned versions for feeding the model
        obs = torch.nan_to_num(obs_raw, nan=0.0)
        fut = torch.nan_to_num(fut_raw, nan=0.0)

        # -------------------------
        # (A) Teacher-forced delta loss
        # -------------------------
        full_raw = torch.cat([obs_raw, fut_raw], dim=1)  # keep NaNs
        gt_delta, mask_delta, full_clean = _make_gt_and_mask(model, full_raw)

        out_tf = model(full_clean)             # (B,T,N,2/4) or (B,T,1,N,2/4)
        pred_mu_tf = _as_delta_pred(out_tf)    # (B,T,N,2)
        pred_delta_tf = pred_mu_tf[:, :-1]     # (B,T-1,N,2)

        delta_loss = ((pred_delta_tf - gt_delta) ** 2 * mask_delta).sum() / mask_delta.sum().clamp_min(1.0)

        # -------------------------
        # (B) Rollout absolute position loss (future only)
        # -------------------------
        fut_valid5 = torch.isfinite(fut_raw).all(dim=-1)            # (B,P,M,5)
        gt_fut6 = model.encoder.add_virtual_root(fut_raw, fut_valid5)  # RAW + mask

        root_valid = fut_valid5.any(dim=-1, keepdim=True)           # (B,P,M,1)
        valid6 = torch.cat([fut_valid5, root_valid], dim=-1)        # (B,P,M,6) root LAST
        mask6 = valid6.unsqueeze(-1).expand(-1, -1, -1, -1, 2).float()

        B, _, M, _, _ = obs.shape

        # Keep BOTH raw and clean windows, plus a validity window
        seq_raw = obs_raw.clone()                                  # (B,O,M,5,2) still has NaNs
        seq_clean = obs.clone()                                    # (B,O,M,5,2) no NaNs
        seq_valid5 = torch.isfinite(seq_raw).all(dim=-1)            # (B,O,M,5)

        preds6 = []
        for _step in range(P):
            out = model(seq_clean)
            mu = _as_delta_pred(out)                               # (B,O,N,2)
            dposN = mu[:, -1]                                      # (B,N,2)
            dpos6 = dposN.view(B, M, 6, 2)                         # (B,M,6,2)

            # last frame validity must come from RAW (not nan_to_num)
            last5_raw = seq_raw[:, -1]                             # (B,M,5,2) may contain NaNs
            last_valid5 = seq_valid5[:, -1]                        # (B,M,5)
            last6 = model.encoder.add_virtual_root(
                last5_raw.unsqueeze(1), last_valid5.unsqueeze(1)
            )[:, 0]                                                # (B,M,6,2)

            next6 = last6 + dpos6                                  # (B,M,6,2)
            next5 = next6[:, :, :5, :]                             # (B,M,5,2) predicted kp are finite
            next_valid5 = torch.ones((B, M, 5), device=device, dtype=torch.bool)

            # recompute root consistently using mask
            next6 = model.encoder.add_virtual_root(
                next5.unsqueeze(1), next_valid5.unsqueeze(1)
            )[:, 0]                                                # (B,M,6,2)

            preds6.append(next6)

            # slide all three windows
            seq_raw = torch.cat([seq_raw[:, 1:], next5.unsqueeze(1)], dim=1)
            seq_valid5 = torch.cat([seq_valid5[:, 1:], next_valid5.unsqueeze(1)], dim=1)
            seq_clean = torch.nan_to_num(seq_raw, nan=0.0)

        pred_fut6 = torch.stack(preds6, dim=1)                     # (B,P,M,6,2)

        pos_loss = (((pred_fut6 - gt_fut6) ** 2) * mask6).sum() / mask6.sum().clamp_min(1.0)

        # -------------------------
        # Combined loss
        # -------------------------
        loss = w_pos * pos_loss + w_delta * delta_loss
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(n, 1)

@torch.no_grad()
def evaluate_one_epoch(
    model, dataloader, device, O, P,
    w_pos=1.0, w_delta=0.5
):
    model.eval()
    total, n = 0.0, 0
    total_pos, total_delta = 0.0, 0.0

    for batch in dataloader:
        obs_raw = batch["obs"].to(device).float()
        fut_raw = batch["fut"].to(device).float()

        obs = torch.nan_to_num(obs_raw, nan=0.0)
        fut = torch.nan_to_num(fut_raw, nan=0.0)

        # ---- teacher-forced delta loss ----
        full_raw = torch.cat([obs_raw, fut_raw], dim=1)
        gt_delta, mask_delta, full_clean = _make_gt_and_mask(model, full_raw)

        out_tf = model(full_clean)
        pred_mu_tf = _as_delta_pred(out_tf)
        pred_delta_tf = pred_mu_tf[:, :-1]

        delta_loss = ((pred_delta_tf - gt_delta) ** 2 * mask_delta).sum() / mask_delta.sum().clamp_min(1.0)

        # ---- rollout pos loss ----
        fut_valid5 = torch.isfinite(fut_raw).all(dim=-1)
        gt_fut6 = model.encoder.add_virtual_root(fut_raw, fut_valid5)

        root_valid = fut_valid5.any(dim=-1, keepdim=True)
        valid6 = torch.cat([fut_valid5, root_valid], dim=-1)
        mask6 = valid6.unsqueeze(-1).expand(-1, -1, -1, -1, 2).float()

        B, _, M, _, _ = obs.shape

        seq_raw = obs_raw.clone()
        seq_clean = obs.clone()
        seq_valid5 = torch.isfinite(seq_raw).all(dim=-1)

        preds6 = []
        for _step in range(P):
            out = model(seq_clean)
            mu = _as_delta_pred(out)
            dposN = mu[:, -1]
            dpos6 = dposN.view(B, M, 6, 2)

            last5_raw = seq_raw[:, -1]
            last_valid5 = seq_valid5[:, -1]
            last6 = model.encoder.add_virtual_root(
                last5_raw.unsqueeze(1), last_valid5.unsqueeze(1)
            )[:, 0]

            next6 = last6 + dpos6
            next5 = next6[:, :, :5, :]
            next_valid5 = torch.ones((B, M, 5), device=device, dtype=torch.bool)

            next6 = model.encoder.add_virtual_root(
                next5.unsqueeze(1), next_valid5.unsqueeze(1)
            )[:, 0]

            preds6.append(next6)

            seq_raw = torch.cat([seq_raw[:, 1:], next5.unsqueeze(1)], dim=1)
            seq_valid5 = torch.cat([seq_valid5[:, 1:], next_valid5.unsqueeze(1)], dim=1)
            seq_clean = torch.nan_to_num(seq_raw, nan=0.0)

        pred_fut6 = torch.stack(preds6, dim=1)
        pos_loss = (((pred_fut6 - gt_fut6) ** 2) * mask6).sum() / mask6.sum().clamp_min(1.0)

        combined = w_pos * pos_loss + w_delta * delta_loss
        if torch.isnan(combined) or torch.isinf(combined):
            continue

        total += float(combined.item())
        total_pos += float(pos_loss.item())
        total_delta += float(delta_loss.item())
        n += 1

    if n == 0:
        return {"loss": float("nan"), "pos_loss": float("nan"), "delta_loss": float("nan")}

    return {"loss": total / n, "pos_loss": total_pos / n, "delta_loss": total_delta / n}


# ----- training -----
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_dl, optimizer, device, O, P, w_pos=0.005, w_delta=1.0)
    eval_stats  = evaluate_one_epoch(model, test_dl, device, O, P, w_pos=0.005, w_delta=1.0)

    print(
        f"epoch {epoch}: "
        f"train={train_loss:.6f}, "
        f"test_total={eval_stats['loss']:.6f}, "
        f"test_pos={eval_stats['pos_loss']:.6f}, "
        f"test_delta={eval_stats['delta_loss']:.6f}"
    )
    wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": eval_stats['loss']})
    
    if (epoch + 1) % SAVE_INTERVAL == 0:
        os.makedirs(f"models/model_weights_{wandb_timestamp}", exist_ok=True)
        torch.save(model.state_dict(), f"models/model_weights_{wandb_timestamp}/epoch{epoch}.pth")