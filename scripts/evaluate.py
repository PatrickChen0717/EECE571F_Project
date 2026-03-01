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
ckpt = "models/model_weights_2026-02-28_18-19-33/epoch86.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

O = 30
P = 30

# ----- dataset -----
paths_left  = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_left.yaml",  recursive=True)
paths_right = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_right.yaml", recursive=True)
yaml_paths = paths_left + paths_right

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
def predict_full_episode_autoreg(model, x_full, O=10, device="cpu", instr_id=0, kp_id=0):
    """
    x_full: (L,M,5,2) absolute positions (NaNs allowed)
    returns pred6: (L,M,6,2)
    """
    model.eval()
    x_full = x_full[..., :2]                         # keep xy
    L, M, K, _ = x_full.shape
    O = min(O, L)

    # keep raw + clean + valid
    x_raw = x_full.clone()
    x_clean = torch.nan_to_num(x_raw, nan=0.0)
    x_valid5 = torch.isfinite(x_raw).all(dim=-1)     # (L,M,5)

    # seed with GT first O frames
    seq_raw   = x_raw[:O].clone()                    # (O,M,5,2)
    seq_clean = x_clean[:O].clone()
    seq_valid = x_valid5[:O].clone()                 # (O,M,5)

    out6 = add_virtual_root_with_model(model, seq_raw, seq_valid)  # (O,M,6,2)
    preds6 = [out6[t].cpu() for t in range(O)]

    for t in range(O, L):
        hist_clean = seq_clean[-O:]                   # (O,M,5,2)
        win_in = hist_clean.unsqueeze(0).to(device)   # (1,O,M,5,2)

        out = model(win_in)                           # (1,O,N,2/4)
        mu = _mu_from_model_out(out)                  # (1,O,N,2)
        dposN = mu[0, -1].detach().cpu()              # (N,2)
        dpos6 = dposN.view(M, 6, 2)                   # (M,6,2)


        last6 = preds6[-1]                            # (M,6,2) on CPU
        next6 = last6 + dpos6                         # (M,6,2)

        next5 = next6[:, :5, :]                       # (M,5,2)
        next_valid5 = torch.ones((M, 5), dtype=torch.bool)  # predicted kps treated as valid
        next6 = add_virtual_root_with_model(model, next5, next_valid5).cpu()

        preds6.append(next6)

        # slide windows (raw validity no longer matters after prediction; predicted are valid)
        seq_clean = torch.cat([seq_clean, next5.unsqueeze(0)], dim=0)

    return torch.stack(preds6, dim=0)                 # (L,M,6,2)

@torch.no_grad()
def plot_full_episode(model, sample, device, instr_id=0, kp_id=0, O=10):
    x_full = torch.as_tensor(sample["x"]).float()
    x_full = x_full[..., :2]

    if x_full.dim() == 3:
        x_full = x_full.unsqueeze(1)  # (L,1,5,2)

    L, M, _, _ = x_full.shape
    valid_xy = torch.isfinite(x_full).all(dim=-1)

    pred6 = predict_full_episode_autoreg(model, x_full, O=O, device=device, instr_id=instr_id, kp_id=kp_id)
    gt_valid5 = torch.isfinite(x_full).all(dim=-1)
    gt6 = add_virtual_root_with_model(model, x_full, gt_valid5)

    node_id = kp_id
    mask = valid_xy[:, instr_id, kp_id].cpu()

    gt_traj = gt6[:, instr_id, node_id]
    pred_traj = pred6[:, instr_id, node_id]

    # ----- split observed vs predicted -----
    O = min(O, L)

    pred_obs = pred_traj[:O]
    pred_fut = pred_traj[O:]

    mask_obs = mask[:O]
    mask_fut = mask[O:]

    plt.figure()

    # Ground truth full trajectory
    plt.plot(gt_traj[mask][:, 0], gt_traj[mask][:, 1],
             color="black", linestyle="--", label="GT Full")

    # Observed frames
    plt.plot(pred_obs[mask_obs][:, 0], pred_obs[mask_obs][:, 1],
             color="blue", label="Observed (O frames)")

    # Predicted future
    plt.plot(pred_fut[mask_fut][:, 0], pred_fut[mask_fut][:, 1],
             color="red", label="Predicted Future")

    plt.axvline(0)  # optional reference
    plt.gca().invert_yaxis()
    plt.title(f"Instr {instr_id}, KP {kp_id} (O={O}, L={L})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- run one sample ----
batch = next(iter(test_dl))
i = 0
sample = {k: (v[i] if torch.is_tensor(v) else v) for k, v in batch.items()}

print("episode length:", sample["x"].shape[0])
plot_full_episode(model, sample, device=device, instr_id=0, kp_id=0, O=O)