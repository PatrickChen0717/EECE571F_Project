import os
import re
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.dinov2_encoder import DINOv2Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

image_dir = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\left_frames"
save_dir = r"C:\Users\Patrick\Downloads\surgmanip_pb_suturing_5hz\dino_features"
os.makedirs(save_dir, exist_ok=True)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

encoder = DINOv2Encoder(
    model_name="facebook/dinov2-small",
    out_dim=128,      # must match training vision_dim
    freeze=True,
    use_cls=False
).to(device)
encoder.eval()

def extract_frame_idx(filename: str) -> int:
    m = re.search(r"frame_(\d+)", filename)
    if not m:
        raise ValueError(f"Cannot parse frame index from {filename}")
    return int(m.group(1))

image_files = []
for fn in os.listdir(image_dir):
    if fn.lower().endswith((".png", ".jpg", ".jpeg")) and "frame_" in fn:
        image_files.append(fn)

image_files.sort()

with torch.no_grad():
    for fn in tqdm(image_files, desc="Precomputing DINO"):
        frame_idx = extract_frame_idx(fn)
        out_path = os.path.join(save_dir, f"{frame_idx:06d}.pt")

        if os.path.exists(out_path):
            continue

        img_path = os.path.join(image_dir, fn)
        img = Image.open(img_path).convert("RGB")
        img = tf(img).unsqueeze(0).to(device)   # (1,3,224,224)

        feat = encoder(img)                     # (1,128) if out_dim=128
        feat = feat.squeeze(0).cpu()           # (128,)

        torch.save(feat, out_path)

print("Done.")