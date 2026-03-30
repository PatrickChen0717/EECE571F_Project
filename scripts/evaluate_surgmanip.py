import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.LSTM import LSTM_gat
from src.model import FullModelWithDINOv2
from data.dataloader_surgmanip import SurgToolSequenceDataset

device = "cpu"

# -----------------------------
# config
# -----------------------------
O = 10
P = 5
BATCH = 1

xml_path = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\left_frames\annotations.xml"
image_dir = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\left_frames"
feature_dir = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\dino_features"

ckpt = r"models\model_weights_2026-03-30_12-48-31\epoch47.pth"

# -----------------------------
# build model
# -----------------------------
encoder = LSTM_gat(hidden_size=128, embed_dim=64)
model = FullModelWithDINOv2(
    encoder,
    vision_dim=128,
    use_visual_diff=True
).to(device)

model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

lp = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Learnable parameters:", lp)

# -----------------------------
# dataset
# -----------------------------
ds = SurgToolSequenceDataset(
    xml_path=xml_path,
    image_dir=image_dir,
    obs_len=O,
    pred_len=P,
    image_transform=None,
    normalize_coords=False,
    include_images=False,
    feature_dir=feature_dir,
    include_features=True,
)

test_dl = DataLoader(ds, batch_size=BATCH, shuffle=False)


# -----------------------------
# helpers
# -----------------------------
def _mu_from_model_out(out):
    """
    out: (B,T,N,2) or (B,T,N,4)
    return: (B,T,N,2)
    """
    if out.ndim != 4:
        raise RuntimeError(f"Unexpected output ndim: {out.ndim}, shape={out.shape}")

    C = out.shape[-1]
    if C == 2:
        return out
    elif C == 4:
        return out[..., :2]
    else:
        raise RuntimeError(f"Unexpected output last dim C={C}")


def add_root_to_coords(coords, vis):
    """
    coords: (..., M, 5, 2)
    vis:    (..., M, 5)

    returns:
      coords6: (..., M, 6, 2)
      vis6:    (..., M, 6)
    """
    w = vis.unsqueeze(-1).float()
    denom = w.sum(dim=-2, keepdim=True).clamp_min(1.0)
    root_xy = (coords * w).sum(dim=-2, keepdim=True) / denom
    root_vis = (vis.sum(dim=-1, keepdim=True) > 0).float()

    coords6 = torch.cat([coords, root_xy], dim=-2)
    vis6 = torch.cat([vis, root_vis], dim=-1)
    return coords6, vis6


def make_delta_input(coords, vis):
    """
    coords: (B,T,M,5,2)
    vis:    (B,T,M,5)

    returns:
      feat: (B,T-1,M,5,3) = [dx, dy, vis_delta]
    """
    delta_xy = coords[:, 1:] - coords[:, :-1]               # (B,T-1,M,5,2)
    valid_delta = vis[:, 1:] * vis[:, :-1]                  # (B,T-1,M,5)
    delta_xy = delta_xy * valid_delta.unsqueeze(-1)
    feat = torch.cat([delta_xy, valid_delta.unsqueeze(-1)], dim=-1)
    return feat


@torch.no_grad()
def rollout_one_sample(model, sample, device="cpu"):
    """
    sample keys from new dataloader:
      obs_coords: (O,10,2)
      obs_vis:    (O,10)
      fut_coords: (P,10,2)
      fut_vis:    (P,10)
      obs_feats:  (O,V)

    returns:
      gt6_full:   (O+P,2,6,2)
      gt6_vis:    (O+P,2,6)
      pred6_full: (O+P,2,6,2)
    """
    obs_coords = sample["obs_coords"].unsqueeze(0).to(device).float()   # (1,O,10,2)
    obs_vis    = sample["obs_vis"].unsqueeze(0).to(device).float()      # (1,O,10)
    fut_coords = sample["fut_coords"].unsqueeze(0).to(device).float()   # (1,P,10,2)
    fut_vis    = sample["fut_vis"].unsqueeze(0).to(device).float()      # (1,P,10)
    obs_feats  = sample["obs_feats"].unsqueeze(0).to(device).float()    # (1,O,V)

    B = 1

    obs_coords = obs_coords.view(B, O, 2, 5, 2)
    obs_vis    = obs_vis.view(B, O, 2, 5)
    fut_coords = fut_coords.view(B, P, 2, 5, 2)
    fut_vis    = fut_vis.view(B, P, 2, 5)

    # GT full sequence
    full_coords = torch.cat([obs_coords, fut_coords], dim=1)   # (1,O+P,2,5,2)
    full_vis    = torch.cat([obs_vis, fut_vis], dim=1)         # (1,O+P,2,5)
    gt6_full, gt6_vis = add_root_to_coords(full_coords, full_vis)

    # start predicted sequence with GT obs block
    obs6, _ = add_root_to_coords(obs_coords, obs_vis)
    preds6 = [obs6[:, t] for t in range(O)]   # list of (1,2,6,2)

    seq_coords = obs_coords.clone()
    seq_vis    = obs_vis.clone()
    seq_feats  = obs_feats.clone()

    for _ in range(P):
        seq_feat = make_delta_input(seq_coords, seq_vis)   # (1,O-1,2,5,3)
        seq_feats_delta = seq_feats[:, 1:]                 # (1,O-1,V)

        out = model(seq_feat, seq_feats_delta)             # (1,O-1,12,2)
        mu = _mu_from_model_out(out)
        dpos6 = mu[:, -1].view(1, 2, 6, 2)                # last step

        last5 = seq_coords[:, -1]                          # (1,2,5,2)
        last5_vis = seq_vis[:, -1]                         # (1,2,5)

        last6, _ = add_root_to_coords(last5, last5_vis)   # (1,2,6,2)

        next6 = last6 + dpos6                              # (1,2,6,2)
        next5 = next6[:, :, :5, :]                         # (1,2,5,2)
        next5_vis = torch.ones((1, 2, 5), device=device)

        preds6.append(next6)

        seq_coords = torch.cat([seq_coords[:, 1:], next5.unsqueeze(1)], dim=1)
        seq_vis    = torch.cat([seq_vis[:, 1:], next5_vis.unsqueeze(1)], dim=1)

        # hold last visual feature for future rollout
        last_feat = seq_feats[:, -1:].clone()
        seq_feats = torch.cat([seq_feats[:, 1:], last_feat], dim=1)

    pred6_full = torch.stack(preds6, dim=1)   # (1,O+P,2,6,2)

    return gt6_full[0].cpu(), gt6_vis[0].cpu(), pred6_full[0].cpu()


def plot_sample_prediction(model, sample, device="cpu", instr_id=0, kp_id=0):
    """
    instr_id: 0 or 1
    kp_id: 0..5  (0..4 are real keypoints, 5 is virtual root)
    """
    gt6_full, gt6_vis, pred6_full = rollout_one_sample(model, sample, device=device)

    gt_traj = gt6_full[:, instr_id, kp_id]        # (O+P,2)
    gt_mask = gt6_vis[:, instr_id, kp_id] > 0.5   # (O+P,)
    pred_traj = pred6_full[:, instr_id, kp_id]    # (O+P,2)

    plt.figure(figsize=(6, 5))

    if gt_mask.any():
        plt.plot(
            gt_traj[gt_mask][:, 0],
            gt_traj[gt_mask][:, 1],
            linestyle="--",
            label="GT Full"
        )

    # observed part
    plt.plot(
        pred_traj[:O, 0],
        pred_traj[:O, 1],
        label="Observed (GT init)"
    )

    # predicted future
    plt.plot(
        pred_traj[O:, 0],
        pred_traj[O:, 1],
        label="Predicted Future"
    )

    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Instr {instr_id}, KP {kp_id}  (O={O}, P={P})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# run one sample
# -----------------------------
def plot_batch_predictions(model, dataset, device="cpu", instr_id=0, kp_id=0, num_samples=8):
    """
    Plot multiple samples in one figure (grid)
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    for i in range(num_samples):
        sample = dataset[i]

        gt6_full, gt6_vis, pred6_full = rollout_one_sample(model, sample, device=device)

        gt_traj = gt6_full[:, instr_id, kp_id]        # (O+P,2)
        gt_mask = gt6_vis[:, instr_id, kp_id] > 0.5
        pred_traj = pred6_full[:, instr_id, kp_id]

        ax = axes[i]

        # GT full trajectory
        if gt_mask.any():
            ax.plot(
                gt_traj[gt_mask][:, 0],
                gt_traj[gt_mask][:, 1],
                linestyle="--",
                color="black"
            )

        # observed part
        ax.plot(
            pred_traj[:O, 0],
            pred_traj[:O, 1],
            color="blue"
        )

        # predicted future
        ax.plot(
            pred_traj[O-1:, 0],
            pred_traj[O-1:, 1],
            color="red"
        )

        ax.set_title(f"Sample {i}")
        ax.invert_yaxis()

    # legend (shared)
    fig.legend(["GT", "Observed", "Predicted"], loc="upper right")
    plt.tight_layout()
    plt.show()
    
plot_batch_predictions(
    model,
    ds,
    device=device,
    instr_id=0,
    kp_id=2,
    num_samples=8
)