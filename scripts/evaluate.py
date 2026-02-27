import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob, random

from src.LSTM import LSTM_gat
from src.model import FullModel
from data.dataloader import KeypointDataset

device = "cuda" if False else "cpu"

# ----- build + load model -----
encoder = LSTM_gat(hidden_size=128, embed_dim=64)
model = FullModel(encoder, gat_out_dim=128).to(device)

ckpt = "models/model_weights_2026-02-26_19-11-03/epoch29.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_params(model)
print("Total params:", f"{total:,}")
print("Trainable params:", f"{trainable:,}")

# ----- dataset -----
paths_left  = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_left.yaml",  recursive=True)
paths_right = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\keypoints_right.yaml", recursive=True)
yaml_paths = paths_left + paths_right

random.seed(42)
n_keep = int(0.05 * len(yaml_paths))
yaml_paths = random.sample(yaml_paths, n_keep)
print("num yamls after 5% sampling:", len(yaml_paths))

ds = KeypointDataset(yaml_paths=yaml_paths, kpt_ids=5, normalize=True)
test_dl = DataLoader(ds, batch_size=32, shuffle=False)

def add_virtual_root_with_model(model, x5):
    """
    x5: (T,5,2) or (B,T,1,5,2)
    returns: (T,6,2) or (B,T,1,6,2)
    """
    if x5.dim() == 3:  # (T,5,2)
        x5 = x5.unsqueeze(0).unsqueeze(2)              # (1,T,1,5,2)
        return model.encoder.add_virtual_root(x5)[0, :, 0]  # (T,6,2)
    else:
        return model.encoder.add_virtual_root(x5)

def _mu_from_model_out(out):
    """
    out can be (B,T,6,2/4) or (B,T,1,6,2/4)
    return mu: (B,T,6,2)
    """
    if out.ndim == 5:
        out = out.squeeze(2)  # (B,T,6,C)
    C = out.shape[-1]
    if C == 2:
        return out
    if C == 4:
        return out[..., 0:2]  # mu only
    raise RuntimeError(f"Unexpected output last dim C={C}")

@torch.no_grad()
def predict_full_episode_autoreg(model, x_full, O=10, device="cpu"):
    """
    x_full: (L,5,2) absolute positions (may contain NaNs)
    returns pred6: (L,6,2)
    """
    model.eval()
    x_full = torch.nan_to_num(x_full, nan=0.0)  # avoid NaN propagation
    L = x_full.shape[0]
    O = min(O, L)

    seq5 = x_full[:O].clone()                              # (O,5,2)
    seq6 = add_virtual_root_with_model(model, seq5)        # (O,6,2)
    out6 = [seq6[i].cpu() for i in range(O)]

    for _ in range(O, L):
        hist5 = seq5[-O:]                                  # (O,5,2)
        win_in = hist5.unsqueeze(0).unsqueeze(2).to(device)  # (1,O,1,5,2)

        out = model(win_in)                                # (1,O,6,C) or (1,O,1,6,C)
        mu = _mu_from_model_out(out)                       # (1,O,6,2)
        dpos6 = mu[0, -1].detach().cpu()                   # (6,2)

        last6 = out6[-1]                                   # (6,2)
        next6 = last6 + dpos6                              # (6,2)

        # rebuild root using the SAME encoder rule (no "mean" assumption)
        next5 = next6[1:].unsqueeze(0)                     # (1,5,2)
        next6 = add_virtual_root_with_model(model, next5)[0].cpu()  # (6,2)

        out6.append(next6)
        seq5 = torch.cat([seq5, next6[1:].unsqueeze(0)], dim=0)     # append predicted 5

    return torch.stack(out6, dim=0)                         # (L,6,2)

@torch.no_grad()
def plot_full_episode(model, sample, device, kp_id=0, O=10):
    x_full = torch.as_tensor(sample["x"]).float()          # (L,5,2)
    valid_xy = torch.isfinite(x_full).all(dim=-1)          # (L,5)

    pred6 = predict_full_episode_autoreg(model, x_full, O=O, device=device)   # (L,6,2)
    gt6   = add_virtual_root_with_model(model, torch.nan_to_num(x_full, nan=0.0))  # (L,6,2)

    node_id = kp_id + 1
    mask = valid_xy[:, kp_id].cpu()

    gt_traj = gt6[:, node_id]
    pred_traj = pred6[:, node_id]

    plt.figure()
    plt.plot(gt_traj[mask][:, 0], gt_traj[mask][:, 1], label="GT")
    plt.plot(pred_traj[mask][:, 0], pred_traj[mask][:, 1], label=f"Pred (autoreg, O={O})")
    plt.gca().invert_yaxis()
    plt.title(f"Full episode trajectory kp={kp_id} (L={x_full.shape[0]})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- run one sample ----
batch = next(iter(test_dl))
i = 0
sample = {k: (v[i] if torch.is_tensor(v) else v) for k, v in batch.items()}

print("episode length:", sample["x"].shape[0])
plot_full_episode(model, sample, device=device, kp_id=0, O=10)