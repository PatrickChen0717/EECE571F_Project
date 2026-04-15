import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tools.helper import compute_ADE, compute_FDE, compute_velocity_error, compute_direction_error, compute_path_length_error, _as_delta_pred, add_virtual_root_from_xy

import glob, random
from src.LSTM import LSTM_gat
from src.transformer import TransformerTrajectoryModel
from src.LSTMonly import LSTMOnlyModel
from src.model import FullModelWithResNet
from src.model import FullModelWithDINOv2
from data.dataloader import KeypointDataset, WindowedKeypointDataset
import pandas as pd
import os

device = "cpu"

# ----- dataset -----
paths_left = glob.glob("/raid/home/patrickbyc/SurgPose_dataset_no_vid/**/keypoints_left.yaml", recursive=True)
yaml_paths = paths_left

random.seed(42)
n_keep = max(1, int(0.25 * len(yaml_paths)))
yaml_paths = random.sample(yaml_paths, n_keep)

ds = KeypointDataset(
    yaml_paths=yaml_paths,
    normalize=False,
    smoothing=False,
    smoothing_window=101,
    load_features=True
)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#############LSTM_GAT############
encoder_hidden_size = 128
encoder_embed_dim = 64
O = 50
P = 10
batch_size = 64
stride = 10

encoder = LSTM_gat(hidden_size=encoder_hidden_size, embed_dim=encoder_embed_dim)
model_LSTMGAT = FullModelWithDINOv2(encoder, vision_dim=128).to(device)

ckpt = "models/model_weights_2026-03-31_19-05-26/epoch45.pth" # 128, 64, 2 GAT, GRU (current best model)
model_LSTMGAT.load_state_dict(torch.load(ckpt, map_location=device))
model_LSTMGAT.eval()

lp = sum(p.numel() for p in model_LSTMGAT.parameters() if p.requires_grad)
print("LSTMGAT Learnable parameters:", lp)

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

ckpt = "models_transformers/model_weights_2026-04-05_12-30-15/epoch54.pth"
model_transformer.load_state_dict(torch.load(ckpt, map_location=device))
model_transformer.eval()
lp = sum(p.numel() for p in model_transformer.parameters() if p.requires_grad)
print("Transformer Learnable parameters:", lp)

#############LSTM############
hidden_size = 128
num_layers = 2

model_LSTM = LSTMOnlyModel(
    M=2,
    hidden_size=hidden_size,
    num_layers=num_layers,
).to(device)


ckpt = "models_lstm/model_weights_2026-04-05_14-20-31/epoch102.pth"
model_LSTM.load_state_dict(torch.load(ckpt, map_location=device))
model_LSTM.eval()
lp = sum(p.numel() for p in model_LSTM.parameters() if p.requires_grad)
print("LSTM Learnable parameters:", lp)

#############################
eval_set = WindowedKeypointDataset(
    ds,
    O=O,
    P=P,
    random_window=False,
    load_from_image=False,
    stride=stride
)

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
            obs_raw = batch["obs"].to(device).float()
            fut_raw = batch["fut"].to(device).float()
            full_frames = batch["full_frames"].to(device).float()

            full_raw = torch.cat([obs_raw, fut_raw], dim=1)

            valid_xy = torch.isfinite(full_raw).all(dim=-1)
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

            fut_valid5 = torch.isfinite(fut_raw).all(dim=-1)
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

metrics_lstmgat, rows_lstmgat = evaluate_one_epoch_local(
    model=model_LSTMGAT,
    dataloader=eval_dl,
    device=device,
    P=P,
    add_root_fn=lambda feat: model_LSTMGAT.encoder.add_virtual_root(feat),
    return_all=True
)

# print("LSTM-GAT Evaluation results:")
# for k, v in metrics_lstmgat.items():
#     print(f"{k}: {v:.6f}")
    
metrics_transformer, rows_transformer = evaluate_one_epoch_local(
    model=model_transformer,
    dataloader=eval_dl,
    device=device,
    P=P,
    add_root_fn=add_virtual_root_from_xy,
    return_all=True
)

# print("Transformer Evaluation results:")
# for k, v in metrics_transformer.items():
#     print(f"{k}: {v:.6f}")
    
metrics_lstm, rows_lstm = evaluate_one_epoch_local(
    model=model_LSTM,
    dataloader=eval_dl,
    device=device,
    P=P,
    add_root_fn=add_virtual_root_from_xy,
    return_all=True
)

# print("LSTM Evaluation results:")
# for k, v in metrics_lstm.items():
#     print(f"{k}: {v:.6f}")
    
print("\n===== Comparison =====")
print(f"LSTM-GAT     vel={metrics_lstmgat['vel_error']:.4f}, dir={metrics_lstmgat['dir_error']:.4f}")
print(f"Transformer  vel={metrics_transformer['vel_error']:.4f}, dir={metrics_transformer['dir_error']:.4f}")
print(f"LSTM         vel={metrics_lstm['vel_error']:.4f}, dir={metrics_lstm['dir_error']:.4f}")


df_lstmgat = pd.DataFrame(rows_lstmgat)
df_transformer = pd.DataFrame(rows_transformer)
df_lstm = pd.DataFrame(rows_lstm)

# ---- Velocity table ----
df_vel = pd.DataFrame({
    "LSTM-GAT": df_lstmgat["vel_error"],
    "Transformer": df_transformer["vel_error"],
    "LSTM": df_lstm["vel_error"],
})
df_vel.to_csv("vel_comparison.csv", index=False)

# ---- Direction table ----
df_dir = pd.DataFrame({
    "LSTM-GAT": df_lstmgat["dir_error"],
    "Transformer": df_transformer["dir_error"],
    "LSTM": df_lstm["dir_error"],
})
df_dir.to_csv("dir_comparison.csv", index=False)


df_pth = pd.DataFrame({
    "LSTM-GAT": df_lstmgat["pth_error"],
    "Transformer": df_transformer["pth_error"],
    "LSTM": df_lstm["pth_error"],
})
df_pth.to_csv("pth_comparison.csv", index=False)


# ---- ADE table ----
df_ade = pd.DataFrame({
    "LSTM-GAT": df_lstmgat["ADE"],
    "Transformer": df_transformer["ADE"],
    "LSTM": df_lstm["ADE"],
})
df_ade.to_csv("ade_comparison.csv", index=False)

# ---- FDE table ----
df_fde = pd.DataFrame({
    "LSTM-GAT": df_lstmgat["FDE"],
    "Transformer": df_transformer["FDE"],
    "LSTM": df_lstm["FDE"],
})
df_fde.to_csv("fde_comparison.csv", index=False)

print("Saved:")
print("  vel_comparison.csv")
print("  dir_comparison.csv")
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