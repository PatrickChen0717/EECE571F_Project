import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from src.LSTM import LSTM_gat
from src.GAT import GAT
from src.model import FullModelWithResNet
from src.model import FullModelWithDINOv2
from data.dataloader_surgmanip import SurgToolSequenceDataset
from torchvision import transforms

from tqdm import tqdm

import glob
import numpy as np

import wandb
wandb.login(key="8b49b325ce8e9e788b2981b63eebbc01ee33bc6b")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- fake dataset example -----
SAVE_INTERVAL = 3
NUM_EPOCHS = 150
BATCH = 64
O = 10
P = 5
M = 2               # Number of instruments
lr = 1e-3
Enable_WandB = True

xml_path = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\left_frames\annotations.xml"
image_dir = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\left_frames"
feature_dir = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\dino_features"

img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

ds = SurgToolSequenceDataset(
    xml_path=xml_path,
    image_dir=image_dir,
    obs_len=O,
    pred_len=P,
    image_transform=img_tf,
    normalize_coords=False,
    include_images=False,
    feature_dir=feature_dir,
    include_features=True,
)

n_total = len(ds)
n_test = int(0.2 * n_total)
n_train = n_total - n_test

train_set, test_set = random_split(
    ds,
    [n_train, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_dl = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_dl  = DataLoader(test_set,  batch_size=BATCH, shuffle=False)
print("num train_set:", len(train_set))
print("num test_set:", len(test_set))



wandb_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
if Enable_WandB:
    job_name = "EECE571F_Project_training" + "dataset: surgmanip" + str(len(xml_path))
    args = {
        "lr": lr,
        "hidden_size": 128,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH,
        "Obs_frames": O,
        "Pred_frames": P,
        "num_keypoint_yaml": len(xml_path),
    }
    wandb.init(
        project="EECE571F_Project",
        name=job_name,
    )
    wandb.config.current_time = wandb_timestamp
    wandb.config.update(args)


# ----- build model -----
encoder = LSTM_gat(hidden_size=128, embed_dim=64)
# model = FullModelWithResNet(encoder, vision_dim=128).to(device)
model = FullModelWithDINOv2(encoder, vision_dim=128).to(device)

optimizer = Adam(model.parameters(), lr=lr)

def _as_delta_pred(model_out):
    """
    model_out can be:
      (B,T,N,2) or (B,T,N,4) or (B,T,1,N,2/4)
    return mu: (B,T,N,2)
    """
    if model_out.ndim == 5:
        model_out = model_out.squeeze(2)
    C = model_out.shape[-1]
    if C == 2:
        return model_out
    elif C == 4:
        return model_out[..., :2]
    else:
        raise RuntimeError(f"Unexpected model output last dim C={C}")

def add_root_to_coords(coords, vis):
    """
    coords: (B,T,M,5,2)
    vis:    (B,T,M,5)

    returns:
      coords6: (B,T,M,6,2)
      vis6:    (B,T,M,6)
    """
    w = vis.unsqueeze(-1)                                         # (B,T,M,5,1)
    denom = w.sum(dim=3, keepdim=True).clamp_min(1.0)
    root_xy = (coords * w).sum(dim=3, keepdim=True) / denom       # (B,T,M,1,2)
    root_vis = (vis.sum(dim=3, keepdim=True) > 0).float()         # (B,T,M,1)

    coords6 = torch.cat([coords, root_xy], dim=3)                 # (B,T,M,6,2)
    vis6 = torch.cat([vis, root_vis], dim=3)                      # (B,T,M,6)
    return coords6, vis6

def make_delta_input(coords, vis):
    """
    coords: (B,T,M,5,2)
    vis:    (B,T,M,5)

    returns:
      feat: (B,T-1,M,5,3)  = [dx, dy, vis_delta]
    """
    delta_xy = coords[:, 1:] - coords[:, :-1]                     # (B,T-1,M,5,2)
    valid_delta = vis[:, 1:] * vis[:, :-1]                        # (B,T-1,M,5)

    delta_xy = delta_xy * valid_delta.unsqueeze(-1)               # zero-out invalid delta
    feat = torch.cat([delta_xy, valid_delta.unsqueeze(-1)], dim=-1)  # (B,T-1,M,5,3)
    return feat

def masked_mse(pred, target, mask):
    """
    pred:   (..., 2)
    target: (..., 2)
    mask:   (...), (...,1), or (...,2)
    """
    se = (pred - target) ** 2

    if mask.ndim == se.ndim - 1:
        mask = mask.unsqueeze(-1)   # (...,1)
    elif mask.ndim != se.ndim:
        raise RuntimeError(f"mask ndim {mask.ndim} incompatible with pred ndim {se.ndim}")

    return (se * mask).sum() / mask.sum().clamp_min(1.0)

def train_one_epoch(
    model, dataloader, optimizer, device, O, P,
    w_pos=1.0, w_delta=0.5, grad_clip=1.0
):
    model.train()
    total, n = 0.0, 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
     
    for batch in pbar:
        obs_coords = batch["obs_coords"].to(device).float()   # (B,O,10,2)
        obs_vis    = batch["obs_vis"].to(device).float()      # (B,O,10)
        fut_coords = batch["fut_coords"].to(device).float()   # (B,P,10,2)
        fut_vis    = batch["fut_vis"].to(device).float()      # (B,P,10)
        obs_feats = batch["obs_feats"].to(device).float()     # (B,O,3,224,224)
        fut_feats = batch["fut_feats"].to(device).float()  
        
        B = obs_coords.shape[0]

        obs_coords = obs_coords.view(B, O, 2, 5, 2)
        obs_vis    = obs_vis.view(B, O, 2, 5)

        fut_coords = fut_coords.view(B, P, 2, 5, 2)
        fut_vis    = fut_vis.view(B, P, 2, 5)

        # ---------- (A) teacher-forced delta loss on observed window ----------
        # obs_feat = make_delta_input(obs_coords, obs_vis)      # (B,O-1,2,5,3)
        # obs_frames_delta = obs_feats[:, 1:]                  # (B,O-1,3,224,224)

        # out_tf = model(obs_feat, obs_frames_delta)
        # pred_delta_tf = _as_delta_pred(out_tf)                # expected (B,O-1,N,2)

        # # GT delta on 6 nodes
        # obs_coords6, obs_vis6 = add_root_to_coords(obs_coords, obs_vis)    # (B,O,2,6,2), (B,O,2,6)
        # gt_delta = obs_coords6[:, 1:] - obs_coords6[:, :-1]                # (B,O-1,2,6,2)
        # gt_delta_mask = (obs_vis6[:, 1:] * obs_vis6[:, :-1]).unsqueeze(-1) # (B,O-1,2,6,1)

        # gt_delta = gt_delta.view(B, O - 1, -1, 2)                          # (B,O-1,12,2)
        # gt_delta_mask = gt_delta_mask.view(B, O - 1, -1, 1)                # (B,O-1,12,1)

        # delta_loss = masked_mse(pred_delta_tf, gt_delta, gt_delta_mask)
        
        full_coords = torch.cat([obs_coords, fut_coords], dim=1)   # (B,O+P,2,5,2)
        full_vis    = torch.cat([obs_vis, fut_vis], dim=1)         # (B,O+P,2,5)
        full_feats  = torch.cat([obs_feats, fut_feats], dim=1)     # (B,O+P,V)

        full_feat_in = make_delta_input(full_coords, full_vis)     # (B,O+P-1,2,5,3)
        full_feats_delta = full_feats[:, 1:]                       # (B,O+P-1,V)

        out_tf = model(full_feat_in, full_feats_delta)
        pred_delta_tf = _as_delta_pred(out_tf)                     # (B,O+P-1,12,2)

        full_coords6, full_vis6 = add_root_to_coords(full_coords, full_vis)
        gt_delta = full_coords6[:, 1:] - full_coords6[:, :-1]      # (B,O+P-1,2,6,2)
        gt_delta_mask = (full_vis6[:, 1:] * full_vis6[:, :-1]).unsqueeze(-1)

        gt_delta = gt_delta.view(B, O + P - 1, -1, 2)             # (B,O+P-1,12,2)
        gt_delta_mask = gt_delta_mask.view(B, O + P - 1, -1, 1)   # (B,O+P-1,12,1)

        delta_loss = masked_mse(pred_delta_tf, gt_delta, gt_delta_mask)

        # ---------- (B) rollout future position loss ----------
        seq_coords = obs_coords.clone()   # (B,O,2,5,2)
        seq_vis    = obs_vis.clone()      # (B,O,2,5)
        seq_frames = obs_feats.clone()   # (B,O,3,224,224)

        preds6 = []

        for _ in range(P):
            seq_feat = make_delta_input(seq_coords, seq_vis)    # (B,O-1,2,5,3)
            seq_frames_delta = seq_frames[:, 1:]                # (B,O-1,3,224,224)

            out = model(seq_feat, seq_frames_delta)
            mu = _as_delta_pred(out)                            # (B,O-1,12,2)
            dpos6 = mu[:, -1].view(B, 2, 6, 2)                 # last predicted step

            last5 = seq_coords[:, -1]                           # (B,2,5,2)
            last5_vis = seq_vis[:, -1]                          # (B,2,5)

            last6, last6_vis = add_root_to_coords(
                last5.unsqueeze(1), last5_vis.unsqueeze(1)
            )
            last6 = last6[:, 0]                                 # (B,2,6,2)

            next6 = last6 + dpos6                               # (B,2,6,2)
            next5 = next6[:, :, :5, :]                          # predicted real keypoints

            # predicted future points are considered visible for autoregressive rollout
            next5_vis = torch.ones((B, 2, 5), device=device, dtype=seq_vis.dtype)

            preds6.append(next6)

            seq_coords = torch.cat([seq_coords[:, 1:], next5.unsqueeze(1)], dim=1)
            seq_vis    = torch.cat([seq_vis[:, 1:], next5_vis.unsqueeze(1)], dim=1)

            # hold last image if no future image is available
            last_frame = seq_frames[:, -1:].clone()
            seq_frames = torch.cat([seq_frames[:, 1:], last_frame], dim=1)

        pred_fut6 = torch.stack(preds6, dim=1)                 # (B,P,2,6,2)

        gt_fut6, gt_fut6_vis = add_root_to_coords(fut_coords, fut_vis)   # (B,P,2,6,2), (B,P,2,6)
        pos_mask = gt_fut6_vis.unsqueeze(-1)                               # (B,P,2,6,1)

        pos_loss = masked_mse(pred_fut6, gt_fut6, pos_mask)

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
@torch.no_grad()
def evaluate_one_epoch(
    model, dataloader, device, O, P,
    w_pos=1.0, w_delta=0.5
):
    model.eval()
    total, n = 0.0, 0
    total_pos, total_delta = 0.0, 0.0

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    
    for batch in pbar:
        obs_coords = batch["obs_coords"].to(device).float()
        obs_vis    = batch["obs_vis"].to(device).float()
        fut_coords = batch["fut_coords"].to(device).float()
        fut_vis    = batch["fut_vis"].to(device).float()
        obs_feats = batch["obs_feats"].to(device).float()
        fut_feats = batch["fut_feats"].to(device).float()
        
        B = obs_coords.shape[0]

        obs_coords = obs_coords.view(B, O, 2, 5, 2)
        obs_vis    = obs_vis.view(B, O, 2, 5)

        fut_coords = fut_coords.view(B, P, 2, 5, 2)
        fut_vis    = fut_vis.view(B, P, 2, 5)

        # teacher-forced delta loss
        # obs_feat = make_delta_input(obs_coords, obs_vis)
        # obs_frames_delta = obs_feats[:, 1:]

        # out_tf = model(obs_feat, obs_frames_delta)
        # pred_delta_tf = _as_delta_pred(out_tf)

        # obs_coords6, obs_vis6 = add_root_to_coords(obs_coords, obs_vis)
        # gt_delta = obs_coords6[:, 1:] - obs_coords6[:, :-1]
        # gt_delta_mask = (obs_vis6[:, 1:] * obs_vis6[:, :-1]).unsqueeze(-1)

        # gt_delta = gt_delta.view(B, O - 1, -1, 2)
        # gt_delta_mask = gt_delta_mask.view(B, O - 1, -1, 1)

        # delta_loss = masked_mse(pred_delta_tf, gt_delta, gt_delta_mask)
        
        full_coords = torch.cat([obs_coords, fut_coords], dim=1)   # (B,O+P,2,5,2)
        full_vis    = torch.cat([obs_vis, fut_vis], dim=1)         # (B,O+P,2,5)
        full_feats  = torch.cat([obs_feats, fut_feats], dim=1)     # (B,O+P,V)
        
        full_feat_in = make_delta_input(full_coords, full_vis)     # (B,O+P-1,2,5,3)
        full_feats_delta = full_feats[:, 1:]                       # (B,O+P-1,V)

        out_tf = model(full_feat_in, full_feats_delta)
        pred_delta_tf = _as_delta_pred(out_tf)                     # (B,O+P-1,12,2)

        full_coords6, full_vis6 = add_root_to_coords(full_coords, full_vis)
        gt_delta = full_coords6[:, 1:] - full_coords6[:, :-1]      # (B,O+P-1,2,6,2)
        gt_delta_mask = (full_vis6[:, 1:] * full_vis6[:, :-1]).unsqueeze(-1)

        gt_delta = gt_delta.view(B, O + P - 1, -1, 2)             # (B,O+P-1,12,2)
        gt_delta_mask = gt_delta_mask.view(B, O + P - 1, -1, 1)   # (B,O+P-1,12,1)

        delta_loss = masked_mse(pred_delta_tf, gt_delta, gt_delta_mask)

        # rollout position loss
        seq_coords = obs_coords.clone()
        seq_vis    = obs_vis.clone()
        seq_frames = obs_feats.clone()

        preds6 = []
        for _ in range(P):
            seq_feat = make_delta_input(seq_coords, seq_vis)
            seq_frames_delta = seq_frames[:, 1:]

            out = model(seq_feat, seq_frames_delta)
            mu = _as_delta_pred(out)
            dpos6 = mu[:, -1].view(B, 2, 6, 2)

            last5 = seq_coords[:, -1]
            last5_vis = seq_vis[:, -1]

            last6, _ = add_root_to_coords(last5.unsqueeze(1), last5_vis.unsqueeze(1))
            last6 = last6[:, 0]

            next6 = last6 + dpos6
            next5 = next6[:, :, :5, :]
            next5_vis = torch.ones((B, 2, 5), device=device, dtype=seq_vis.dtype)

            preds6.append(next6)

            seq_coords = torch.cat([seq_coords[:, 1:], next5.unsqueeze(1)], dim=1)
            seq_vis    = torch.cat([seq_vis[:, 1:], next5_vis.unsqueeze(1)], dim=1)

            last_frame = seq_frames[:, -1:].clone()
            seq_frames = torch.cat([seq_frames[:, 1:], last_frame], dim=1)

        pred_fut6 = torch.stack(preds6, dim=1)

        gt_fut6, gt_fut6_vis = add_root_to_coords(fut_coords, fut_vis)
        pos_mask = gt_fut6_vis.unsqueeze(-1)

        pos_loss = masked_mse(pred_fut6, gt_fut6, pos_mask)

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
    eval_stats = evaluate_one_epoch(model, test_dl, device, O, P, w_pos=0.005, w_delta=1.0)

    print(
        f"epoch {epoch}: "
        f"train={train_loss:.6f}, "
        f"test_total={eval_stats['loss']:.6f}, "
        f"test_pos={eval_stats['pos_loss']:.6f}, "
        f"test_delta={eval_stats['delta_loss']:.6f}"
    )

    if Enable_WandB:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": eval_stats["loss"],
            "test_pos_loss": eval_stats["pos_loss"],
            "test_delta_loss": eval_stats["delta_loss"],
        })

    if (epoch + 1) % SAVE_INTERVAL == 0:
        os.makedirs(f"models/model_weights_{wandb_timestamp}", exist_ok=True)
        torch.save(model.state_dict(), f"models/model_weights_{wandb_timestamp}/epoch{epoch}.pth")