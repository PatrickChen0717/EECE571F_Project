import os

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from data.dataloader_surgmanip import SurgToolSequenceDataset
from src.LSTM import LSTM_gat
from src.model import FullModelWithDINOv2

load_dotenv()

ENCODER_HIDDEN_SIZE = 128
ENCODER_EMBED_DIM = 64
NUM_TOOLS = 2
OBS_LEN = 50
PRED_LEN = 10
BATCH_SIZE = 64


def _mu_from_model_out(out):
    if out.ndim == 5:
        out = out.squeeze(2)

    channels = out.shape[-1]
    if channels == 2:
        return out
    if channels == 4:
        return out[..., :2]
    raise RuntimeError(f"Unexpected output last dim C={channels}")


def add_virtual_root_with_model(model, x5, valid5=None):
    if x5.dim() == 3:
        x5_bt = x5.unsqueeze(0).unsqueeze(0)
        valid5_bt = (
            torch.isfinite(x5_bt).all(dim=-1)
            if valid5 is None
            else valid5.unsqueeze(0).unsqueeze(0)
        )
        x5_clean = torch.nan_to_num(x5_bt, nan=0.0)
        feat_bt = torch.cat([x5_clean, valid5_bt.unsqueeze(-1).float()], dim=-1)
        return model.encoder.add_virtual_root(feat_bt)[0, 0, ..., :2]

    if x5.dim() == 4:
        x5_bt = x5.unsqueeze(0)
        valid5_bt = (
            torch.isfinite(x5_bt).all(dim=-1) if valid5 is None else valid5.unsqueeze(0)
        )
        x5_clean = torch.nan_to_num(x5_bt, nan=0.0)
        feat_bt = torch.cat([x5_clean, valid5_bt.unsqueeze(-1).float()], dim=-1)
        return model.encoder.add_virtual_root(feat_bt)[0, ..., :2]

    if x5.dim() == 5:
        if valid5 is None:
            valid5 = torch.isfinite(x5).all(dim=-1)
        x5_clean = torch.nan_to_num(x5, nan=0.0)
        feat = torch.cat([x5_clean, valid5.unsqueeze(-1).float()], dim=-1)
        return model.encoder.add_virtual_root(feat)[..., :2]

    raise ValueError(f"Unexpected x5 dim: {x5.dim()}")


def _get_grouped_sample_tensors(sample, prefix, device):
    coords_key = f"{prefix}_coords_by_tool"
    vis_key = f"{prefix}_vis_by_tool"
    if coords_key in sample and vis_key in sample:
        return sample[coords_key].to(device).float(), sample[vis_key].to(device).float()

    coords = sample[f"{prefix}_coords"].to(device).float()
    vis = sample[f"{prefix}_vis"].to(device).float()
    if coords.ndim != 3 or vis.ndim != 2:
        raise RuntimeError(
            f"Expected flat {prefix} tensors with shapes (T,10,2) and (T,10); "
            f"got {tuple(coords.shape)} and {tuple(vis.shape)}"
        )

    timesteps, num_points, channels = coords.shape
    if channels != 2 or num_points % NUM_TOOLS != 0:
        raise RuntimeError(
            f"Cannot reshape {prefix}_coords with shape {tuple(coords.shape)} into "
            f"(T,{NUM_TOOLS},K,2)"
        )

    num_keypoints = num_points // NUM_TOOLS
    return (
        coords.view(timesteps, NUM_TOOLS, num_keypoints, 2),
        vis.view(timesteps, NUM_TOOLS, num_keypoints),
    )


def _get_obs_feature_sequence(sample, device):
    if "obs_delta_feats" in sample:
        return sample["obs_delta_feats"].to(device).float()

    if "obs_feats" in sample:
        return sample["obs_feats"][1:].to(device).float()

    raise KeyError("Expected `obs_delta_feats` or `obs_feats` in sample.")


def _build_delta_with_vis(seq_raw, valid5):
    seq_clean = torch.nan_to_num(seq_raw, nan=0.0)
    delta_xy = seq_clean[1:] - seq_clean[:-1]
    valid_delta = valid5[1:] & valid5[:-1]
    delta_xy = torch.where(valid_delta.unsqueeze(-1), delta_xy, torch.zeros_like(delta_xy))
    return torch.cat([delta_xy, valid_delta.unsqueeze(-1).float()], dim=-1)


@torch.no_grad()
def rollout_window(model, sample, device):
    obs_raw, obs_vis = _get_grouped_sample_tensors(sample, "obs", device)
    seq_feats = _get_obs_feature_sequence(sample, device)

    seq_raw = obs_raw.clone()
    seq_valid5 = obs_vis > 0.5

    preds6 = []
    for _ in range(PRED_LEN):
        seq_delta = _build_delta_with_vis(seq_raw, seq_valid5)
        out = model(seq_delta.unsqueeze(0), seq_feats.unsqueeze(0))
        mu = _mu_from_model_out(out)
        dpos6 = mu[0, -1].view(NUM_TOOLS, 6, 2)

        last5_raw = seq_raw[-1]
        last_valid5 = seq_valid5[-1]
        last6 = add_virtual_root_with_model(model, last5_raw, last_valid5)

        next6 = last6 + dpos6
        next5 = next6[:, :5, :]
        next_valid5 = torch.ones((NUM_TOOLS, 5), device=device, dtype=torch.bool)
        next6 = add_virtual_root_with_model(model, next5, next_valid5)

        preds6.append(next6.detach().cpu())

        seq_raw = torch.cat([seq_raw[1:], next5.unsqueeze(0)], dim=0)
        seq_valid5 = torch.cat([seq_valid5[1:], next_valid5.unsqueeze(0)], dim=0)
        # Match training-time rollout by reusing the last observed feature vector.
        seq_feats = torch.cat([seq_feats[1:], seq_feats[-1:]], dim=0)

    return torch.stack(preds6, dim=0)


@torch.no_grad()
def plot_window_rollout(
    model, sample, device, sample_num, save_directory, instr_id=0, kp_id=0
):
    obs_raw, obs_vis = _get_grouped_sample_tensors(sample, "obs", device)
    fut_raw, fut_vis = _get_grouped_sample_tensors(sample, "fut", device)

    full_raw = torch.cat([obs_raw, fut_raw], dim=0)
    full_vis = torch.cat([obs_vis, fut_vis], dim=0) > 0.5

    obs6 = add_virtual_root_with_model(model, obs_raw, obs_vis > 0.5).cpu()
    gt6 = add_virtual_root_with_model(model, full_raw, full_vis).cpu()
    pred_fut6 = rollout_window(model, sample, device)

    gt_traj = gt6[:, instr_id, kp_id]
    obs_traj = obs6[:, instr_id, kp_id]
    pred_traj = pred_fut6[:, instr_id, kp_id]

    gt_mask = full_vis[:, instr_id, kp_id].cpu()
    obs_mask = (obs_vis[:, instr_id, kp_id] > 0.5).cpu()

    plt.figure()

    if gt_mask.any():
        plt.plot(
            gt_traj[gt_mask][:, 0],
            gt_traj[gt_mask][:, 1],
            color="black",
            linestyle="--",
            label="GT Window",
        )

    if obs_mask.any():
        plt.plot(
            obs_traj[obs_mask][:, 0],
            obs_traj[obs_mask][:, 1],
            color="blue",
            label="Observed",
        )

    plt.plot(
        pred_traj[:, 0],
        pred_traj[:, 1],
        color="red",
        marker="o",
        label="Predicted Future",
    )

    plt.gca().invert_yaxis()
    plt.title(
        f"Sample {sample_num}, Instr {instr_id}, KP {kp_id} "
        f"(O={OBS_LEN}, P={PRED_LEN})"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(
        save_directory, f"traj_sample{sample_num}_instr{instr_id}_kp{kp_id}.png"
    )
    print(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()


def _sample_from_batch(batch, index):
    sample = {}
    for key, value in batch.items():
        sample[key] = value[index] if torch.is_tensor(value) else value[index]
    return sample


def run_through_all_samples(model, dataloader, device, save_directory):
    sample_num = 0

    for batch in dataloader:
        batch_size_current = next(iter(batch.values())).shape[0]
        for batch_index in range(batch_size_current):
            sample = _sample_from_batch(batch, batch_index)
            for instr_id in range(NUM_TOOLS):
                for kp_id in range(5):
                    plot_window_rollout(
                        model,
                        sample,
                        device=device,
                        sample_num=sample_num,
                        save_directory=save_directory,
                        instr_id=instr_id,
                        kp_id=kp_id,
                    )
            sample_num += 1


def build_model(device, checkpoint_path):
    encoder = LSTM_gat(
        hidden_size=ENCODER_HIDDEN_SIZE,
        embed_dim=ENCODER_EMBED_DIM,
    )
    model = FullModelWithDINOv2(encoder, vision_dim=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def build_dataloader():
    dataset_dir = os.getenv("SURGMANIP_DIR")
    if dataset_dir is None:
        raise ValueError("SURGMANIP_DIR is not set.")

    dataset = SurgToolSequenceDataset(
        dataset_dir=dataset_dir,
        split="val",
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        image_transform=None,
        normalize_coords=False,
        include_images=False,
        include_features=True,
    )
    print("dataset size:", len(dataset))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.getenv("BASE_DIR")
    if base_dir is None:
        raise ValueError("BASE_DIR is not set.")

    checkpoint_path = os.path.join(base_dir, "models", "epoch29.pth")
    save_directory = os.path.join(base_dir, "outputs", "fine_plots2")

    model = build_model(device, checkpoint_path)
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Learnable parameters:", learnable_params)

    dataloader = build_dataloader()
    run_through_all_samples(model, dataloader, device, save_directory)


if __name__ == "__main__":
    main()
