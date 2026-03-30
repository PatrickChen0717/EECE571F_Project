import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.dinov2_encoder import DINOv2Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

root = r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset"

paths_left = glob.glob(os.path.join(root, "**", "regular", "left_frames"), recursive=True)
print("found left_frames dirs:", len(paths_left))
for p in paths_left[:5]:
    print("  ", p)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

encoder = DINOv2Encoder(
    model_name="facebook/dinov2-small",
    out_dim=128,
    freeze=True,
    use_cls=False
).to(device)
encoder.eval()


def extract_frame_idx(filename: str) -> int:
    stem = os.path.splitext(filename)[0]   # "000001"
    return int(stem)


with torch.no_grad():
    for image_dir in paths_left:
        print(f"\nProcessing folder: {image_dir}")

        save_dir = os.path.join(os.path.dirname(image_dir), "left_frame_pt")
        os.makedirs(save_dir, exist_ok=True)
        print("save_dir:", save_dir)

        image_files = [
            fn for fn in os.listdir(image_dir)
            if fn.lower().endswith(".png")
        ]
        image_files.sort(key=lambda fn: int(os.path.splitext(fn)[0]))

        print("num png files:", len(image_files))
        if len(image_files) > 0:
            print("first few:", image_files[:3])

        for fn in tqdm(image_files, desc="Precomputing DINO", leave=False):
            frame_idx = extract_frame_idx(fn)
            out_path = os.path.join(save_dir, f"{frame_idx:06d}.pt")

            if os.path.exists(out_path):
                continue

            img_path = os.path.join(image_dir, fn)
            img = Image.open(img_path).convert("RGB")
            img = tf(img).unsqueeze(0).to(device)

            feat = encoder(img).squeeze(0).cpu()
            torch.save(feat, out_path)

print("Done.")