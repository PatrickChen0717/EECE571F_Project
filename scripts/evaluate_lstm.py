import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob, random

from src.LSTMonly import LSTMOnlyModel
from data.dataloader import KeypointDataset, WindowedKeypointDataset
from PIL import Image
from torchvision import transforms
import os

device = "cpu"
hidden_size = 128
num_layers = 2

O = 50
P = 10
batch_size = 64

model = LSTMOnlyModel(
    M=2,
    hidden_size=hidden_size,
    num_layers=num_layers,
).to(device)


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

ckpt = "models_lstm/model_weights_2026-04-05_12-30-15/epoch54.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

lp = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Learnable parameters:", lp)

paths_left = glob.glob(
    "/raid/home/patrickbyc/SurgPose_dataset_no_vid/**/keypoints_left.yaml",
    recursive=True
)
yaml_paths = paths_left

print("num yamls found:", len(yaml_paths))

random.seed(42)
n_keep = max(1, int(0.25 * len(yaml_paths)))
yaml_paths = random.sample(yaml_paths, n_keep)

print("num yamls kept:", len(yaml_paths))

ds = KeypointDataset(
    yaml_paths=yaml_paths,
    normalize=False,
    smoothing=True,
    smoothing_window=101,
    load_features=True
)
print("dataset size:", len(ds))

test_dl = DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_frame_path(episode_path, frame_idx):
    if isinstance(episode_path, list):
        episode_path = episode_path[0]

    episode_dir = os.path.dirname(episode_path)

    candidates = [
        os.path.join(episode_dir, "regular", "left_frames", f"{frame_idx:06d}.png"),
        os.path.join(episode_dir, "left_frames", f"{frame_idx:06d}.png"),
        os.path.join(episode_dir, f"left_{frame_idx:06d}.png"),
        os.path.join(episode_dir, f"{frame_idx:06d}.png"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"No frame found for frame_idx={frame_idx} under {episode_dir}"
    )


def load_frame_tensor(episode_path, frame_idx, device="cpu"):
    frame_path = get_frame_path(episode_path, frame_idx)
    img = Image.open(frame_path).convert("RGB")
    frame = img_transform(img).unsqueeze(0).to(device)   # (1,3,224,224)
    return frame

def load_feature_tensor(episode_path, frame_idx, device="cpu"):
    if isinstance(episode_path, list):
        episode_path = episode_path[0]

    episode_dir = os.path.dirname(episode_path)

    feat_path = os.path.join(
        episode_dir,
        "regular/left_frame_pt",
        f"{frame_idx:06d}.pt"
    )

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature not found: {feat_path}")

    feat = torch.load(feat_path, map_location=device)   # (D,)
    return feat.unsqueeze(0)   # (1,D)


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

def add_virtual_root_with_model(model, x5, valid5=None):
    """
    Accepts:
      (M,5,2)
      (T,M,5,2)
      (B,T,M,5,2)

    valid5 matches leading dims:
      (M,5), (T,M,5), or (B,T,M,5)

    Returns:
      (M,6,2) or (T,M,6,2) or (B,T,M,6,2)
    """
    if x5.dim() == 3:
        x5_bt = x5.unsqueeze(0).unsqueeze(0)  # (1,1,M,5,2)
        if valid5 is None:
            valid5_bt = torch.isfinite(x5_bt).all(dim=-1)
        else:
            valid5_bt = valid5.unsqueeze(0).unsqueeze(0)
        x5_clean = torch.nan_to_num(x5_bt, nan=0.0)
        feat_bt = torch.cat([x5_clean, valid5_bt.unsqueeze(-1).float()], dim=-1)  # (1,1,M,5,3)
        x6_bt = add_virtual_root_from_xy(feat_bt)[..., :2]  # (1,1,M,6,2)
        return x6_bt[0, 0]

    elif x5.dim() == 4:
        x5_bt = x5.unsqueeze(0)  # (1,T,M,5,2)
        if valid5 is None:
            valid5_bt = torch.isfinite(x5_bt).all(dim=-1)
        else:
            valid5_bt = valid5.unsqueeze(0)
        x5_clean = torch.nan_to_num(x5_bt, nan=0.0)
        feat_bt = torch.cat([x5_clean, valid5_bt.unsqueeze(-1).float()], dim=-1)  # (1,T,M,5,3)
        x6_bt = add_virtual_root_from_xy(feat_bt)[..., :2]  # (1,T,M,6,2)
        return x6_bt[0]

    elif x5.dim() == 5:
        if valid5 is None:
            valid5 = torch.isfinite(x5).all(dim=-1)
        x5_clean = torch.nan_to_num(x5, nan=0.0)
        feat = torch.cat([x5_clean, valid5.unsqueeze(-1).float()], dim=-1)  # (B,T,M,5,3)
        return add_virtual_root_from_xy(feat)[..., :2]

    else:
        raise ValueError(f"Unexpected x5 dim: {x5.dim()}")


def build_delta_with_vis(seq_raw):
    """
    seq_raw: (T,M,5,2) or (B,T,M,5,2), may contain NaN

    returns:
      delta_in: (...,T-1,M,5,3) = [dx, dy, valid]
      seq_clean
      valid5
    """
    valid5 = torch.isfinite(seq_raw).all(dim=-1)
    seq_clean = torch.nan_to_num(seq_raw, nan=0.0)

    delta_xy = seq_clean[1:] - seq_clean[:-1] if seq_raw.dim() == 4 else seq_clean[:, 1:] - seq_clean[:, :-1]
    valid_delta = valid5[1:] & valid5[:-1] if seq_raw.dim() == 4 else valid5[:, 1:] & valid5[:, :-1]

    delta_xy = torch.where(valid_delta.unsqueeze(-1), delta_xy, torch.zeros_like(delta_xy))
    delta_in = torch.cat([delta_xy, valid_delta.unsqueeze(-1).float()], dim=-1)
    return delta_in, seq_clean, valid5


@torch.no_grad()
def predict_episode_blockwise_no_overlap(model, x_full, episode_path, O=30, P=10, device="cpu"):
    """
    Blockwise evaluation with NO overlap:
      block k uses GT obs [t:t+O) -> predicts [t+O:t+O+P)
      then t += O+P

    x_full: (L,M,5,2) or (L,5,2), may contain NaN
    returns:
      pred6: (L,M,6,2) with NaNs where not filled
    """
    model.eval()

    x_full = torch.as_tensor(x_full, dtype=torch.float32)
    x_full = x_full[..., :2]

    if x_full.dim() == 3:
        x_full = x_full.unsqueeze(1)  # (L,1,5,2)

    L, M, K, C = x_full.shape
    if K != 5 or C != 2:
        raise ValueError(f"Expected x_full shape (L,M,5,2), got {tuple(x_full.shape)}")

    pred6 = torch.full((L, M, 6, 2), float("nan"), dtype=torch.float32)

    def gt5_to_6(x5):
        valid5 = torch.isfinite(x5).all(dim=-1)
        return add_virtual_root_with_model(model, x5, valid5)

    t = 0
    stride = O + P

    while t + O <= L:
        obs5_raw = x_full[t:t + O].clone()   # (O,M,5,2)
        obs6 = gt5_to_6(obs5_raw)            # (O,M,6,2)
        pred6[t:t + O] = obs6.cpu()

        if t + O >= L:
            break

        frame_idx = t + O - 1

        feat = load_feature_tensor(episode_path, frame_idx, device=device)  # (1,D)

        seq_raw = obs5_raw.clone()    # (O,M,5,2), may contain NaN
        last6 = obs6[-1].cpu()        # (M,6,2)

        maxP = min(P, L - (t + O))
        for k in range(maxP):
            win_delta, _, _ = build_delta_with_vis(seq_raw)   # (O-1,M,5,3)
            win_in = win_delta.unsqueeze(0).to(device)        # (1,O-1,M,5,3)

            T = win_in.shape[1]
            feat_seq = feat.unsqueeze(1).repeat(1, T, 1)      # (1,T,D)

            out = model(win_in, feat_seq)
            mu = _mu_from_model_out(out)                      # (1,T,N,2) usually
            dposN = mu[0, -1].detach().cpu()                  # (N,2)

            if dposN.numel() != M * 6 * 2:
                raise RuntimeError(
                    f"Model output size mismatch: got {tuple(dposN.shape)}, "
                    f"cannot reshape to ({M}, 6, 2)"
                )

            dpos6 = dposN.view(M, 6, 2)

            next6 = last6 + dpos6           # (M,6,2)
            next5 = next6[:, :5, :]         # (M,5,2)
            next_valid5 = torch.isfinite(next5).all(dim=-1)

            next6 = add_virtual_root_with_model(model, next5, next_valid5).cpu()

            pred6[t + O + k] = next6
            last6 = next6

            seq_raw = torch.cat([seq_raw[1:], next5.unsqueeze(0)], dim=0)

        t += stride

    return pred6


@torch.no_grad()
def plot_full_episode(model, sample, device, sample_num, instr_id=0, kp_id=0, O=10, save_directory=None):
    x_full = torch.as_tensor(sample["x"]).float()
    x_full = x_full[..., :2]

    if x_full.dim() == 3:
        x_full = x_full.unsqueeze(1)

    L, M, _, _ = x_full.shape
    valid_xy = torch.isfinite(x_full).all(dim=-1)

    pred6 = predict_episode_blockwise_no_overlap(
        model,
        x_full,
        episode_path=sample["episode_path"],
        O=O,
        P=P,
        device=device
    )

    gt_valid5 = torch.isfinite(x_full).all(dim=-1)
    gt6 = add_virtual_root_with_model(model, x_full, gt_valid5)

    node_id = kp_id
    mask = valid_xy[:, instr_id, kp_id].cpu()

    gt_traj = gt6[:, instr_id, node_id]
    pred_traj = pred6[:, instr_id, node_id]

    O = min(O, L)

    plt.figure()

    plt.plot(
        gt_traj[mask][:, 0],
        gt_traj[mask][:, 1],
        color="black",
        linestyle="--",
        label="GT Full"
    )

    stride = O + P
    L_pred = pred_traj.shape[0]

    shown_obs = False
    shown_pred = False

    for t in range(0, L_pred, stride):
        obs_start = t
        obs_end = min(t + O, L_pred)

        obs_block = pred_traj[obs_start:obs_end]
        obs_mask = mask[obs_start:obs_end]

        if obs_mask.any():
            plt.plot(
                obs_block[obs_mask][:, 0],
                obs_block[obs_mask][:, 1],
                color="blue",
                label="Observed (O)" if not shown_obs else None
            )
            shown_obs = True

        pred_start = obs_end
        pred_end = min(pred_start + P, L_pred)

        fut_block = pred_traj[pred_start:pred_end]
        fut_mask = torch.isfinite(fut_block).all(dim=-1)

        if fut_mask.any():
            plt.plot(
                fut_block[fut_mask][:, 0],
                fut_block[fut_mask][:, 1],
                color="red",
                label="Predicted (P)" if not shown_pred else None
            )
            shown_pred = True

    plt.gca().invert_yaxis()
    plt.title(f"Sample {sample_num}, Instr {instr_id}, KP {kp_id} (O={O}, P={P})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    
    if save_directory is None:
        save_dir = "/raid/home/patrickbyc/EECE571F_Project/outputs/lstm_plots"
    else:
        save_dir = save_directory
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"traj_sample{sample_num}_instr{instr_id}_kp{kp_id}.png")
    print(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()

def run_through_all_sample(num_samples):
    batch = next(iter(test_dl))
    save_dir = "/raid/home/patrickbyc/EECE571F_Project/outputs/lstm_fine_plots"
    
    for i in range(num_samples):
        sample = {}
        for k, v in batch.items():
            sample[k] = v[i]

        for instr_id in range(2):
            for kp_id in range(5):
                plot_full_episode(model, sample, device=device, sample_num=i, instr_id=instr_id, kp_id=kp_id, O=O, save_directory=save_dir)

run_through_all_sample(len(yaml_paths))
