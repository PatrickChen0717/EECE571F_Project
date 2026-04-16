import os
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from tools.helper import compute_ADE, compute_FDE, compute_velocity_error, compute_direction_error, compute_path_length_error, _as_delta_pred, add_virtual_root_from_xy

from src.LSTM import LSTM_gat
from src.transformer import TransformerTrajectoryModel
from src.LSTMonly import LSTMOnlyModel
from src.model import FullModelWithDINOv2
from data.dataloader_surgmanip import SurgToolSequenceDataset
import pandas as pd

device = "cpu"

DEFAULT_DATASET_DIR = Path("/home/lycpaul/Dataset/surgmanip/dataset")
if not DEFAULT_DATASET_DIR.exists():
    DEFAULT_DATASET_DIR = Path(os.getenv("SURGMANIP_DIR")) if os.getenv("SURGMANIP_DIR") else None

DATA_SPLIT_SEED = 42


def prepare_model(model, ckpt, label):
    if ckpt is None:
        print(f"{label}: checkpoint is None, skipping evaluation.")
        return False

    checkpoint = torch.load(ckpt, map_location=device)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    lp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{label} Learnable parameters:", lp)
    return True

def _resolve_sequence_dirs(dataset_dir: Path) -> list[Path]:
    sequence_dirs = []
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "instances.xml").exists() or (entry / "annotations.xml").exists():
            sequence_dirs.append(entry)

    if not sequence_dirs:
        raise FileNotFoundError(
            f"No sequence folders with annotations found in {dataset_dir}"
        )

    return sequence_dirs


def _resolve_annotation_path(sequence_dir: Path) -> Path:
    for annotation_name in ("instances.xml", "annotations.xml"):
        candidate = sequence_dir / annotation_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Missing annotation XML in {sequence_dir}")


def _split_sequence_dirs(
    sequence_dirs: list[Path], train_ratio: float = 0.8, seed: int = DATA_SPLIT_SEED
) -> tuple[list[Path], list[Path]]:
    num_sequences = len(sequence_dirs)
    if num_sequences < 2:
        raise ValueError("Need at least two sequences to create train/val splits.")

    num_train = max(1, int(train_ratio * num_sequences))
    if num_train >= num_sequences:
        num_train = num_sequences - 1

    permutation = torch.randperm(
        num_sequences, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train_indices = permutation[:num_train]
    val_indices = permutation[num_train:]

    train_dirs = [sequence_dirs[idx] for idx in train_indices]
    val_dirs = [sequence_dirs[idx] for idx in val_indices]
    return train_dirs, val_dirs


def _build_sequence_dataset(sequence_dir: Path, obs_len: int, pred_len: int):
    return SurgToolSequenceDataset(
        xml_path=str(_resolve_annotation_path(sequence_dir)),
        image_dir=str(sequence_dir),
        feature_dir=str(sequence_dir),
        obs_len=obs_len,
        pred_len=pred_len,
        image_transform=None,
        normalize_coords=False,
        include_images=False,
        include_features=True,
    )


#############LSTM_GAT############
encoder_hidden_size = 128
encoder_embed_dim = 64
O = 50
P = 10
batch_size = 64
stride = 10

encoder = LSTM_gat(hidden_size=encoder_hidden_size, embed_dim=encoder_embed_dim)
model_LSTMGAT = FullModelWithDINOv2(encoder, vision_dim=128).to(device)

# ckpt = "models/model_weights_2026-03-31_19-05-26/epoch45.pth" # 128, 64, 2 GAT, GRU (current best model)
ckpt = None
has_lstmgat = prepare_model(model_LSTMGAT, ckpt, "LSTMGAT")

#############Transformer############
vision_dim=128
d_model=128
nhead=4
num_layers=2
ff_dim=256
dropout=0.1

model_transformer = TransformerTrajectoryModel(
    M=2,
    vision_dim=vision_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    ff_dim=ff_dim,
    dropout=dropout,
).to(device)

ckpt = "models_transformers/model_weights_2026-04-15_13-17-06/epoch6.pth"
has_transformer = prepare_model(model_transformer, ckpt, "Transformer")

#############LSTM############
hidden_size = 128
num_layers = 2

model_LSTM = LSTMOnlyModel(
    M=2,
    hidden_size=hidden_size,
    num_layers=num_layers,
).to(device)


# ckpt = "models_lstm/model_weights_2026-04-05_14-20-31/epoch102.pth"
ckpt = None
has_lstm = prepare_model(model_LSTM, ckpt, "LSTM")

#############################
dataset_dir = DEFAULT_DATASET_DIR
if dataset_dir is None:
    raise ValueError("Dataset directory is not set.")
dataset_dir = dataset_dir.expanduser().resolve()

sequence_dirs = _resolve_sequence_dirs(dataset_dir)
_, val_sequence_dirs = _split_sequence_dirs(sequence_dirs)

eval_set = ConcatDataset(
    [_build_sequence_dataset(sequence_dir, O, P) for sequence_dir in val_sequence_dirs]
)

print(f"Dataset directory: {dataset_dir}")
print(
    f"Eval sequences ({len(val_sequence_dirs)}): "
    f"{[path.name for path in val_sequence_dirs]}"
)
print(f"Eval samples: {len(eval_set)}")

eval_dl = DataLoader(
    eval_set,
    batch_size=batch_size,
    shuffle=False
)

def evaluate_one_epoch_local(model, dataloader, device, P, add_root_fn, return_all=False):
    model.eval()

    total_vel, total_dir = 0.0, 0.0
    total_ade, total_fde = 0.0, 0.0
    total_pth = 0.0
    n = 0
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            obs_raw = batch["obs_coords_by_tool"].to(device).float()
            obs_vis = batch["obs_vis_by_tool"].to(device).float()
            fut_raw = batch["fut_coords_by_tool"].to(device).float()
            fut_vis = batch["fut_vis_by_tool"].to(device).float()
            full_frames = batch["full_feats"].to(device).float()

            full_raw = torch.cat([obs_raw, fut_raw], dim=1)
            full_vis = torch.cat([obs_vis, fut_vis], dim=1)

            valid_xy = full_vis > 0.5
            pos_clean = torch.nan_to_num(full_raw, nan=0.0)

            delta_xy = pos_clean[:, 1:] - pos_clean[:, :-1]
            valid_delta = valid_xy[:, 1:] & valid_xy[:, :-1]
            delta_xy = torch.where(
                valid_delta.unsqueeze(-1),
                delta_xy,
                torch.zeros_like(delta_xy)
            )

            full_delta_in = torch.cat(
                [delta_xy, valid_delta.unsqueeze(-1).float()],
                dim=-1
            )

            out_tf = model(full_delta_in, full_frames)
            pred_delta = _as_delta_pred(out_tf)
            pred_fut6 = pred_delta[:, -P:]   # (B,P,N,2)

            fut_valid5 = fut_vis > 0.5
            fut_clean = torch.nan_to_num(fut_raw, nan=0.0)

            fut_feat = torch.cat(
                [fut_clean, fut_valid5.unsqueeze(-1).float()],
                dim=-1
            )   # (B,P,M,5,3)

            gt_fut6 = add_root_fn(fut_feat)[..., :2]
            gt_fut6 = gt_fut6.reshape(gt_fut6.size(0), gt_fut6.size(1), -1, 2)

            B = pred_fut6.size(0)

            for i in range(B):
                pred_i = pred_fut6[i:i+1]
                gt_i = gt_fut6[i:i+1]

                vel_err = compute_velocity_error(pred_i, gt_i).item()
                dir_err = compute_direction_error(pred_i, gt_i).item()
                pth_err = compute_path_length_error(pred_i, gt_i).item()
                ade = compute_ADE(pred_i, gt_i).item()
                fde = compute_FDE(pred_i, gt_i).item()

                total_vel += vel_err
                total_dir += dir_err
                total_pth += pth_err
                total_ade += ade
                total_fde += fde
                n += 1

                if return_all:
                    rows.append({
                        "vel_error": vel_err,
                        "dir_error": dir_err,
                        "pth_error": pth_err,
                        "ADE": ade,
                        "FDE": fde,
                    })

    metrics = {
        "vel_error": total_vel / n if n > 0 else float("nan"),
        "dir_error": total_dir / n if n > 0 else float("nan"),
        "pth_error": total_pth / n if n > 0 else float("nan"),
        "ADE": total_ade / n if n > 0 else float("nan"),
        "FDE": total_fde / n if n > 0 else float("nan"),
    }

    if return_all:
        return metrics, rows
    return metrics

results = {}

if has_lstmgat:
    results["LSTM-GAT"] = evaluate_one_epoch_local(
        model=model_LSTMGAT,
        dataloader=eval_dl,
        device=device,
        P=P,
        add_root_fn=lambda feat: model_LSTMGAT.encoder.add_virtual_root(feat),
        return_all=True,
    )

if has_transformer:
    results["Transformer"] = evaluate_one_epoch_local(
        model=model_transformer,
        dataloader=eval_dl,
        device=device,
        P=P,
        add_root_fn=add_virtual_root_from_xy,
        return_all=True,
    )

if has_lstm:
    results["LSTM"] = evaluate_one_epoch_local(
        model=model_LSTM,
        dataloader=eval_dl,
        device=device,
        P=P,
        add_root_fn=add_virtual_root_from_xy,
        return_all=True,
    )

print("\n===== Comparison =====")

if results:
    for label, (metrics, _) in results.items():
        print(f"{label:<12} vel={metrics['vel_error']:.4f}, dir={metrics['dir_error']:.4f}")

    row_frames = {
        label: pd.DataFrame(rows)
        for label, (_, rows) in results.items()
    }

    df_vel = pd.DataFrame({
        label: df["vel_error"]
        for label, df in row_frames.items()
    })
    df_vel.to_csv("vel_comparison.csv", index=False)

    df_dir = pd.DataFrame({
        label: df["dir_error"]
        for label, df in row_frames.items()
    })
    df_dir.to_csv("dir_comparison.csv", index=False)

    df_pth = pd.DataFrame({
        label: df["pth_error"]
        for label, df in row_frames.items()
    })
    df_pth.to_csv("pth_comparison.csv", index=False)

    df_ade = pd.DataFrame({
        label: df["ADE"]
        for label, df in row_frames.items()
    })
    df_ade.to_csv("ade_comparison.csv", index=False)

    df_fde = pd.DataFrame({
        label: df["FDE"]
        for label, df in row_frames.items()
    })
    df_fde.to_csv("fde_comparison.csv", index=False)

    print("Saved:")
    print("  vel_comparison.csv")
    print("  dir_comparison.csv")
    print("  pth_comparison.csv")
    print("  ade_comparison.csv")
    print("  fde_comparison.csv")

    print("\n===== Averages =====")

    print("Velocity:")
    print(df_vel.mean())

    print("\nDirection:")
    print(df_dir.mean())

    print("\nPath:")
    print(df_pth.mean())

    print("\nADE:")
    print(df_ade.mean())

    print("\nFDE:")
    print(df_fde.mean())
else:
    print("No metrics were computed because all checkpoints are None.")