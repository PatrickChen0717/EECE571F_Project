import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob, random

from src.LSTM import LSTM_gat
from src.model import FullModel
from data.dataloader import KeypointDataset

device = "cpu"

# ----- build + load model -----
encoder = LSTM_gat(hidden_size=128, embed_dim=64)
model = FullModel(encoder, gat_out_dim=128).to(device)

# ckpt = "models/model_weights_2026-02-27_23-57-36/epoch35.pth"
ckpt = "models/model_weights_2026-03-09_22-50-50/epoch74.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

O = 30
P = 30

# ----- dataset -----
paths_left  = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_left.yaml",  recursive=True)
# paths_right = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_right.yaml", recursive=True)
yaml_paths = paths_left

random.seed(42)
n_keep = int(0.05*len(yaml_paths))
yaml_paths = random.sample(yaml_paths, n_keep)

ds = KeypointDataset(yaml_paths=yaml_paths, normalize=False)
test_dl = DataLoader(ds, batch_size=32, shuffle=False)

def _mu_from_model_out(out):
    # out: (B,T,N,2/4) or (B,T,1,N,2/4)
    if out.ndim == 5:
        out = out.squeeze(2)
    C = out.shape[-1]
    if C == 2:
        return out
    if C == 4:
        return out[..., 0:2]
    raise RuntimeError(f"Unexpected output last dim C={C}")

def add_virtual_root_with_model(model, x5, valid5=None):
    """
    Accepts:
      (M,5,2)
      (T,M,5,2)
      (B,T,M,5,2)
    valid5 matches the leading dims:
      (M,5) or (T,M,5) or (B,T,M,5)
    Returns:
      (M,6,2) or (T,M,6,2) or (B,T,M,6,2)
    """
    if x5.dim() == 3:
        # (M,5,2) -> (1,1,M,5,2)
        x5_bt = x5.unsqueeze(0).unsqueeze(0)
        if valid5 is None:
            valid5_bt = torch.isfinite(x5_bt).all(dim=-1)  # (1,1,M,5)
        else:
            valid5_bt = valid5.unsqueeze(0).unsqueeze(0)
        x6_bt = model.encoder.add_virtual_root(x5_bt, valid5_bt)   # (1,1,M,6,2)
        return x6_bt[0, 0]

    elif x5.dim() == 4:
        # (T,M,5,2) -> (1,T,M,5,2)
        x5_bt = x5.unsqueeze(0)
        if valid5 is None:
            valid5_bt = torch.isfinite(x5_bt).all(dim=-1)          # (1,T,M,5)
        else:
            valid5_bt = valid5.unsqueeze(0)
        x6_bt = model.encoder.add_virtual_root(x5_bt, valid5_bt)   # (1,T,M,6,2)
        return x6_bt[0]

    elif x5.dim() == 5:
        # (B,T,M,5,2)
        if valid5 is None:
            valid5 = torch.isfinite(x5).all(dim=-1)
        return model.encoder.add_virtual_root(x5, valid5)

    else:
        raise ValueError(f"Unexpected x5 dim: {x5.dim()}")


@torch.no_grad()
def predict_episode_blockwise_no_overlap(model, x_full, O=30, P=10, device="cpu"):
    """
    Blockwise evaluation with NO overlap:
      block k uses GT obs [t:t+O) -> predicts [t+O:t+O+P)
      then t += O+P (next block starts after the predicted segment)

    x_full: (L,M,5,2) absolute positions (NaNs allowed)
    returns pred6: (L,M,6,2) with NaNs where not filled
    """
    model.eval()

    x_full = x_full[..., :2]
    if x_full.dim() == 3:
        x_full = x_full.unsqueeze(1)  # (L,1,5,2)

    L, M, _, _ = x_full.shape
    pred6 = torch.full((L, M, 6, 2), float("nan"))

    # Helper: build GT 6-node with correct masked root (root LAST)
    def gt5_to_6(x5):
        # x5: (T,M,5,2) possibly NaNs
        valid5 = torch.isfinite(x5).all(dim=-1)             # (T,M,5)
        x6 = add_virtual_root_with_model(model, x5, valid5)  # (T,M,6,2)
        return x6

    t = 0
    stride = O + P  # <-- NO overlap blocks

    while t + O <= L:
        # ----- GT observation window -----
        obs5_raw = x_full[t:t+O].clone()                    # (O,M,5,2) NaNs allowed
        obs6 = gt5_to_6(obs5_raw)                            # (O,M,6,2)
        pred6[t:t+O] = obs6.cpu()

        # if no room to predict future, stop
        if t + O >= L:
            break

        # ----- autoregressive prediction for P steps -----
        seq_clean = torch.nan_to_num(obs5_raw, nan=0.0)      # (O,M,5,2)
        seq_valid5 = torch.isfinite(obs5_raw).all(dim=-1)    # (O,M,5)
        last6 = obs6[-1].cpu()                               # (M,6,2)

        maxP = min(P, L - (t + O))
        for k in range(maxP):
            win_delta = seq_clean[1:] - seq_clean[:-1]       # (O-1,M,5,2)
            win_delta_valid = seq_valid5[1:] & seq_valid5[:-1]
            win_delta = torch.where(
                win_delta_valid.unsqueeze(-1),
                win_delta,
                torch.zeros_like(win_delta)
            )

            win_in = win_delta.unsqueeze(0).to(device)       # (1,O-1,M,5,2)

            out = model(win_in)
            mu = _mu_from_model_out(out)
            dposN = mu[0, -1].detach().cpu()
            dpos6 = dposN.view(M, 6, 2)

            next6 = last6 + dpos6
            next5 = next6[:, :5, :]
            next_valid5 = torch.ones((M, 5), dtype=torch.bool)

            next6 = add_virtual_root_with_model(model, next5, next_valid5).cpu()

            pred6[t+O+k] = next6
            last6 = next6

            seq_clean = torch.cat([seq_clean[1:], next5.unsqueeze(0)], dim=0)
            seq_valid5 = torch.cat([seq_valid5[1:], next_valid5.unsqueeze(0)], dim=0)

        t += stride

    return pred6

@torch.no_grad()
def plot_full_episode(model, sample, device, instr_id=0, kp_id=0, O=10):
    x_full = torch.as_tensor(sample["x"]).float()
    x_full = x_full[..., :2]

    if x_full.dim() == 3:
        x_full = x_full.unsqueeze(1)  # (L,1,5,2)

    L, M, _, _ = x_full.shape
    valid_xy = torch.isfinite(x_full).all(dim=-1)

    pred6 = predict_episode_blockwise_no_overlap(model, x_full, O=O, P=P, device=device)
    gt_valid5 = torch.isfinite(x_full).all(dim=-1)
    gt6 = add_virtual_root_with_model(model, x_full, gt_valid5)

    node_id = kp_id
    mask = valid_xy[:, instr_id, kp_id].cpu()

    gt_traj = gt6[:, instr_id, node_id]
    pred_traj = pred6[:, instr_id, node_id]

    # ----- split observed vs predicted -----
    O = min(O, L)

    plt.figure()

    # Ground truth (full)
    plt.plot(gt_traj[mask][:, 0], gt_traj[mask][:, 1],
            color="black", linestyle="--", label="GT Full")

    stride = O + P
    L_pred = pred_traj.shape[0]

    for t in range(0, L_pred, stride):
        # ---- Observation block ----
        obs_start = t
        obs_end   = min(t + O, L_pred)

        obs_block = pred_traj[obs_start:obs_end]
        obs_mask  = mask[obs_start:obs_end]

        plt.plot(obs_block[obs_mask][:, 0],
                obs_block[obs_mask][:, 1],
                color="blue")

        # ---- Prediction block ----
        pred_start = obs_end
        pred_end   = min(pred_start + P, L_pred)

        fut_block = pred_traj[pred_start:pred_end]
        fut_mask  = mask[pred_start:pred_end]

        plt.plot(fut_block[fut_mask][:, 0],
                fut_block[fut_mask][:, 1],
                color="red")

    plt.gca().invert_yaxis()
    plt.title(f"Instr {instr_id}, KP {kp_id} (O={O}, P={P})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["GT Full", "Observed (O)", "Predicted (P)"])
    plt.tight_layout()
    plt.show()

# ---- run one sample ----
batch = next(iter(test_dl))
i = 0
sample = {k: (v[i] if torch.is_tensor(v) else v) for k, v in batch.items()}

print("episode length:", sample["x"].shape[0])
plot_full_episode(model, sample, device=device, instr_id=0, kp_id=1, O=O)