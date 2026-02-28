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

ckpt = "models/model_weights_2026-02-27_19-30-26/epoch14.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

O = 80
P = 40

# ----- dataset -----
paths_left  = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_left.yaml",  recursive=True)
# paths_right = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_right.yaml", recursive=True)
yaml_paths = paths_left

random.seed(42)
n_keep = int(0.05 * len(yaml_paths))
yaml_paths = random.sample(yaml_paths, n_keep)

ds = KeypointDataset(yaml_paths=yaml_paths, normalize=True)
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

def add_virtual_root_with_model(model, x5):
    """
    x5:
      - (T,M,5,2) or (B,T,M,5,2)
    returns:
      - (T,M,6,2) or (B,T,M,6,2)
    """
    if x5.dim() == 4:
        # (T,M,5,2) -> (1,T,M,5,2)
        x5_b = x5.unsqueeze(0)
        x6_b = model.encoder.add_virtual_root(x5_b)   # (1,T,M,6,2)
        return x6_b[0]                                # (T,M,6,2)
    elif x5.dim() == 5:
        return model.encoder.add_virtual_root(x5)
    else:
        raise ValueError(f"Unexpected x5 dim: {x5.dim()}")

@torch.no_grad()
def predict_full_episode_autoreg(model, x_full, O=10, device="cpu", instr_id=0, kp_id=0):
    """
    x_full: (L,M,5,2) absolute positions (NaNs allowed)
    returns pred6: (L,M,6,2)
    """
    model.eval()
    x_full = x_full[..., :2]
    x_full = torch.nan_to_num(x_full, nan=0.0)
    L, M, K, _ = x_full.shape
    O = min(O, L)

    # seed with GT first O frames
    seq = x_full[:O].clone()                         # (O,M,5,2)
    out6 = add_virtual_root_with_model(model, seq)   # (O,M,6,2)
    preds6 = [out6[t].cpu() for t in range(O)]       # list of (M,6,2)

    for _ in range(O, L):
        hist = seq[-O:]                               # (O,M,5,2)
        win_in = hist.unsqueeze(0).to(device)         # (1,O,M,5,2)
        # print("step", _, "hist_last(kp0)", hist[-1, 0, 0].tolist()) 
        
        out = model(win_in)                           # (1,O,N,C)
        mu = _mu_from_model_out(out)                  # (1,O,N,2)
        dposN = mu[0, -1].detach().cpu()              # (N,2) where N=M*6
        
        # reshape N back to (M,6,2)
        dpos6 = dposN.view(M, 6, 2)                   # (M,6,2)
        # print("step", _, "delta", dpos6[instr_id, kp_id])
        
        last6 = preds6[-1]                            # (M,6,2)
        next6 = last6 + dpos6                         # (M,6,2)

        # enforce root consistency using the SAME encoder rule
        next5 = next6[:, :5, :]                      # (M,5,2)
        next6 = add_virtual_root_with_model(model, next5.unsqueeze(0))[0].cpu()  # (M,6,2)

        preds6.append(next6)
        seq = torch.cat([seq, next5.unsqueeze(0)], dim=0)  # (t,M,5,2)

    # print("Last absolute position:")
    # print(last6)

    # print("Next predicted absolute position:")
    # print(next6)

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
    gt6   = add_virtual_root_with_model(model, torch.nan_to_num(x_full, nan=0.0))

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