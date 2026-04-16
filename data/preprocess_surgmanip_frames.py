import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract all frames from an MP4 into a train/val folder."
    )
    parser.add_argument("video_path", type=Path, help="Path to the input MP4 video.")
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--train", action="store_true", help="Save frames under parent_dir/train."
    )
    split_group.add_argument(
        "--val", action="store_true", help="Save frames under parent_dir/val."
    )
    return parser.parse_args()


def extract_frames(video_path: Path, split_name: str) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an MP4 file, got: {video_path}")

    video_file_name = video_path.name.split(".")[0]

    output_dir = video_path.parent / ".." / split_name / video_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = output_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames were extracted from: {video_path}")

    print(f"Saved {frame_idx} frames to {output_dir}")
    return output_dir


def main():
    args = parse_args()
    split_name = "train" if args.train else "val"
    extract_frames(args.video_path, split_name)


if __name__ == "__main__":
    main()
