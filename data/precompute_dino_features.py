import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from src.dinov2_encoder import DINOv2Encoder
from dotenv import load_dotenv

load_dotenv()

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DINO embeddings for image folders under a train/val split."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=os.getenv("SURGMANIP_DIR"),
        help="Dataset root. Defaults to the SURGMANIP_DIR environment variable.",
    )
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--train",
        action="store_true",
        help="Process image folders under dataset_dir/train.",
    )
    split_group.add_argument(
        "--val",
        action="store_true",
        help="Process image folders under dataset_dir/val.",
    )
    return parser.parse_args()


def get_split_dir(args) -> Path:
    if args.dataset_dir is None:
        raise ValueError(
            "dataset_dir is not set. Pass --dataset-dir or set SURGMANIP_DIR."
        )

    split_name = "train" if args.train else "val"
    split_dir = Path(args.dataset_dir) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return split_dir


def contains_images(directory: Path) -> bool:
    return any(
        child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
        for child in directory.iterdir()
    )


def find_image_dirs(split_dir: Path) -> list[Path]:
    image_dirs = [
        path for path in split_dir.rglob("*") if path.is_dir() and contains_images(path)
    ]
    return sorted(image_dirs)


def sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def main():
    args = parse_args()
    split_dir = get_split_dir(args)
    image_dirs = find_image_dirs(split_dir)

    print(f"Scanning image directories under {split_dir}")
    print(f"Found {len(image_dirs)} directories containing images")

    encoder = DINOv2Encoder(
        model_name="facebook/dinov2-small", out_dim=128, freeze=True, use_cls=False
    ).to(device)
    encoder.eval()

    with torch.no_grad():
        for image_dir in image_dirs:
            print(f"\nProcessing folder: {image_dir}")
            image_files = sorted(
                [
                    path
                    for path in image_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ],
                key=sort_key,
            )

            print("num image files:", len(image_files))
            if image_files:
                print("first few:", [path.name for path in image_files[:3]])

            for image_path in tqdm(image_files, desc="Precomputing DINO", leave=False):
                out_path = image_dir / f"{image_path.stem}.pt"
                if out_path.exists():
                    continue

                img = Image.open(image_path).convert("RGB")
                img = tf(img).unsqueeze(0).to(device)

                feat = encoder(img).squeeze(0).cpu()
                torch.save(feat, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
