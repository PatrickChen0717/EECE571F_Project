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
wandb.login(key="")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- fake dataset example -----
SAVE_INTERVAL = 3
NUM_EPOCHS = 30
BATCH = 32
O = 10
P = 5
M = 2               # Number of instruments
lr = 5e-4
Enable_WandB = True

paths_left = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_left.yaml", recursive=True)
paths_right = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_right.yaml", recursive=True)

yaml_paths = paths_left + paths_right

print("num yamls:", len(yaml_paths))

if len(yaml_paths) == 0:
    raise RuntimeError("No YAML files found. Check the path and recursive glob.")

ds = KeypointDataset(
    yaml_paths=yaml_paths,
    kpt_ids=5,
    normalize=True
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
    if model_out.ndim == 5:
        model_out = model_out.squeeze(2)  # (B,T,6,C)
    C = model_out.shape[-1]
    if C == 2:
        return model_out
    elif C == 4:
        return model_out[..., :2]
    else:
        raise RuntimeError(f"Unexpected model output last dim C={C}")

def train_one_epoch(model, dataloader, optimizer, device, O, P, grad_clip=1.0):
    model.train()
    total_loss, n_batches = 0.0, 0

    start = O - 1
    end = O - 1 + P

    for batch in dataloader:
        obs = batch["obs"].to(device).float()   # (B,O,5,2)
        fut = batch["fut"].to(device).float()   # (B,P,5,2)
        pos_raw = torch.cat([obs, fut], dim=1)  # (B,O+P,5,2)
        valid_xy = torch.isfinite(pos_raw).all(dim=-1)  # (B,T,5)

        pos = torch.nan_to_num(pos_raw, nan=0.0).unsqueeze(2)  # (B,T,1,5,2)

        # ----- GT delta in 6-node space -----
        pos6 = model.encoder.add_virtual_root(pos)             # (B,T,1,6,2)
        gt_delta = (pos6[:, 1:] - pos6[:, :-1]).squeeze(2)     # (B,T-1,6,2)

        # ----- Pred delta -----
        out = model(pos)                                       # (B,T,6,2/4) or (B,T,1,6,2/4)
        pred_mu = _as_delta_pred(out)                          # (B,T,6,2)
        pred_delta = pred_mu[:, :-1]                           # (B,T-1,6,2)

        # ----- Mask for valid deltas -----
        valid_delta5 = valid_xy[:, 1:] & valid_xy[:, :-1]      # (B,T-1,5)
        root_valid_t  = valid_xy.any(dim=2)                    # (B,T)
        root_valid_dt = root_valid_t[:, 1:] & root_valid_t[:, :-1]  # (B,T-1)

        valid_delta6 = torch.cat([valid_delta5, root_valid_dt.unsqueeze(-1)], dim=2)  # (B,T-1,6)
        mask = valid_delta6.unsqueeze(-1).expand(-1, -1, -1, 2).float()               # (B,T-1,6,2)

        # ----- only supervise future horizon (P steps) -----
        pred_future = pred_delta[:, start:end]   # (B,P,6,2)
        gt_future   = gt_delta[:, start:end]     # (B,P,6,2)
        mask_future = mask[:, start:end]         # (B,P,6,2)

        # masked MSE
        diff2 = (pred_future - gt_future) ** 2
        loss = (diff2 * mask_future).sum() / mask_future.sum().clamp_min(1.0)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in dataloader:
        obs = batch["obs"].to(device).float()   # (B,O,5,2)
        fut = batch["fut"].to(device).float()   # (B,P,5,2)
        pos_raw = torch.cat([obs, fut], dim=1)  # (B,O+P,5,2)
        pos = torch.nan_to_num(pos_raw, nan=0.0).unsqueeze(2)

        # ----- build mask -----
        valid_xy = torch.isfinite(pos_raw).all(dim=-1)
        valid_delta5 = valid_xy[:, 1:] & valid_xy[:, :-1]
        root_valid_t  = valid_xy.any(dim=2)
        root_valid_dt = root_valid_t[:, 1:] & root_valid_t[:, :-1]
        valid_delta6 = torch.cat([valid_delta5, root_valid_dt.unsqueeze(-1)], dim=2)
        valid_delta6 = valid_delta6.unsqueeze(-1).expand(-1, -1, -1, 2).float()

        # ----- GT -----
        pos6 = model.encoder.add_virtual_root(pos)
        gt_delta = (pos6[:, 1:] - pos6[:, :-1]).squeeze(2)

        # ----- pred -----
        out = model(pos)
        pred_mu = _as_delta_pred(out)
        pred_delta = pred_mu[:, :-1]

        # ----- O/P slicing -----
        start = O - 1
        end = O - 1 + P

        pred_future = pred_delta[:, start:end]
        gt_future   = gt_delta[:, start:end]
        mask_future = valid_delta6[:, start:end]

        loss = ((pred_future - gt_future)**2 * mask_future).sum() / mask_future.sum().clamp_min(1.0)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)

# ----- training -----
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_dl, optimizer, device, O, P)
    test_loss  = evaluate_one_epoch(model, test_dl, device)

    print(f"epoch {epoch}: train={train_loss:.6f}, test={test_loss:.6f}")
    wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
    
    if (epoch + 1) % SAVE_INTERVAL == 0:
        os.makedirs(f"models/model_weights_{wandb_timestamp}", exist_ok=True)
        torch.save(model.state_dict(), f"models/model_weights_{wandb_timestamp}/epoch{epoch}.pth")