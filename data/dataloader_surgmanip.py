import os
import re
import math
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

load_dotenv()

SURGMANIP_DIR = os.getenv("SURGMANIP_DIR")
TRAIN_DIR = os.path.join(SURGMANIP_DIR, "train") if SURGMANIP_DIR else None
VAL_DIR = os.path.join(SURGMANIP_DIR, "val") if SURGMANIP_DIR else None


LABEL_ORDER = ["shaft", "wrist", "ee", "tip1", "tip2"]
NUM_TOOL_KPTS = 5
NUM_TOOLS = 2
NUM_KEYPOINTS = NUM_TOOL_KPTS * NUM_TOOLS  # 10


def parse_frame_number(name: str) -> int:
    """
    Extract integer frame number from strings like:
      frame_000024
      surgmanip_cn_suturing_30hz_frame_000024.png
      000024.png
    """
    stem = os.path.splitext(os.path.basename(name))[0]

    for pattern in (r"frame_(\d+)$", r"(\d+)$"):
        m = re.search(pattern, stem)
        if m:
            return int(m.group(1))

    raise ValueError(f"Cannot parse frame index from name: {name}")


def point_distance(p1: Optional[np.ndarray], p2: Optional[np.ndarray]) -> float:
    if p1 is None or p2 is None:
        return float("inf")
    return float(np.linalg.norm(p1 - p2))


def safe_float(s: str) -> float:
    return float(s.strip())


def parse_points_string(points_str: str) -> np.ndarray:
    """
    CVAT point format: 'x,y'
    """
    x_str, y_str = points_str.split(",")
    return np.array([safe_float(x_str), safe_float(y_str)], dtype=np.float32)


def load_cvat_points_xml(xml_path: str) -> List[Dict]:
    """
    Parse CVAT XML.

    Returns a list of frame dicts:
    [
      {
        "frame_idx": int,
        "width": int,
        "height": int,
        "points_by_label": {
            "shaft": [{"xy": np.array([x,y]), "occluded": 0 or 1}, ...],
            ...
        }
      },
      ...
    ]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames = []
    for image_el in root.findall("image"):
        frame_name = image_el.attrib["name"]
        frame_idx = parse_frame_number(frame_name)
        width = int(image_el.attrib["width"])
        height = int(image_el.attrib["height"])

        points_by_label = {lab: [] for lab in LABEL_ORDER}

        for pt_el in image_el.findall("points"):
            label = pt_el.attrib.get("label")
            if label not in points_by_label:
                continue

            xy = parse_points_string(pt_el.attrib["points"])
            occluded = int(pt_el.attrib.get("occluded", "0"))

            points_by_label[label].append(
                {
                    "xy": xy,
                    "occluded": occluded,
                    "instance": pt_el.attrib.get("instance"),
                }
            )

        frames.append(
            {
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "width": width,
                "height": height,
                "points_by_label": points_by_label,
            }
        )

    frames.sort(key=lambda x: x["frame_idx"])
    return frames


def build_tool_centers_from_partial(
    frame_coords: np.ndarray, frame_vis: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    frame_coords: (10,2)
    frame_vis:    (10,)
    Return approximate centers for left and right tools from available keypoints.
    """
    left_pts = []
    right_pts = []

    for i in range(NUM_TOOL_KPTS):
        if frame_vis[i] > 0.5:
            left_pts.append(frame_coords[i])
    for i in range(NUM_TOOL_KPTS, NUM_KEYPOINTS):
        if frame_vis[i] > 0.5:
            right_pts.append(frame_coords[i])

    left_center = (
        np.mean(left_pts, axis=0).astype(np.float32) if len(left_pts) > 0 else None
    )
    right_center = (
        np.mean(right_pts, axis=0).astype(np.float32) if len(right_pts) > 0 else None
    )
    return left_center, right_center


def convert_frame_to_fixed_layout(frame_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one frame into fixed 10-keypoint layout.

    Order:
      left:  shaft,wrist,ee,tip1,tip2
      right: shaft,wrist,ee,tip1,tip2

    Rule:
    - instance="1" -> left
    - instance="2" -> right
    - if instance is missing, fall back to the legacy x-based assignment
    - occluded=1 => keep coord but vis=0
    """
    coords = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    vis = np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
    assigned = np.zeros((NUM_KEYPOINTS,), dtype=bool)
    assigned_occ = np.full((NUM_KEYPOINTS,), 2, dtype=np.int32)

    img_mid_x = frame_dict["width"] / 2.0

    def set_slot(idx: int, xy: np.ndarray, occluded: int) -> None:
        # Prefer visible annotations if duplicate points are present for a slot.
        if assigned[idx] and assigned_occ[idx] <= occluded:
            return

        coords[idx] = xy
        vis[idx] = 0.0 if occluded == 1 else 1.0
        assigned[idx] = True
        assigned_occ[idx] = occluded

    for label_idx, label in enumerate(LABEL_ORDER):
        items = frame_dict["points_by_label"][label]
        left_idx = label_idx
        right_idx = NUM_TOOL_KPTS + label_idx

        fallback_pts = []
        for item in items:
            instance = item.get("instance")
            if instance == "1":
                set_slot(left_idx, item["xy"], item["occluded"])
            elif instance == "2":
                set_slot(right_idx, item["xy"], item["occluded"])
            else:
                fallback_pts.append((item["xy"], item["occluded"]))

        if len(fallback_pts) >= 2:
            fallback_pts = sorted(fallback_pts, key=lambda t: t[0][0])
            left_pt, left_occ = fallback_pts[0]
            right_pt, right_occ = fallback_pts[-1]

            if not assigned[left_idx]:
                set_slot(left_idx, left_pt, left_occ)
            if not assigned[right_idx]:
                set_slot(right_idx, right_pt, right_occ)

        elif len(fallback_pts) == 1:
            pt, occ = fallback_pts[0]

            if pt[0] < img_mid_x and not assigned[left_idx]:
                set_slot(left_idx, pt, occ)
            elif pt[0] >= img_mid_x and not assigned[right_idx]:
                set_slot(right_idx, pt, occ)
            elif not assigned[left_idx]:
                set_slot(left_idx, pt, occ)
            elif not assigned[right_idx]:
                set_slot(right_idx, pt, occ)

    return coords, vis


def forward_fill_observation_window(
    obs_coords: np.ndarray, obs_vis: np.ndarray
) -> np.ndarray:
    """
    Only fill the observation window before feeding into the model.
    Keep target frames untouched.

    obs_coords: (O,10,2)
    obs_vis:    (O,10)
    returns filled_obs_coords: (O,10,2)
    """
    filled = obs_coords.copy()
    O, N, _ = filled.shape

    for n in range(N):
        last_valid = None
        for t in range(O):
            if obs_vis[t, n] > 0.5:
                last_valid = filled[t, n].copy()
            else:
                if last_valid is not None:
                    filled[t, n] = last_valid
                # else leave as zero
    return filled


def smooth_valid_trajectory(coords, vis, window=5):
    """
    coords: (T,10,2)
    vis:    (T,10)
    returns smoothed coords: (T,10,2)

    Smooth each keypoint over time, only within contiguous valid segments.
    """
    T, N, D = coords.shape
    out = coords.copy()

    radius = window // 2

    for n in range(N):
        valid = vis[:, n] > 0.5

        start = None
        for t in range(T + 1):
            if t < T and valid[t]:
                if start is None:
                    start = t
            else:
                if start is not None:
                    end = t  # segment [start, end)
                    seg = coords[start:end, n]  # (L,2)
                    L = end - start

                    if L >= window:
                        seg_sm = seg.copy()
                        for i in range(L):
                            l = max(0, i - radius)
                            r = min(L, i + radius + 1)
                            seg_sm[i] = seg[l:r].mean(axis=0)
                        out[start:end, n] = seg_sm

                    start = None

    return out


class SurgToolSequenceDataset(Dataset):
    """
    Dataset for either:
      - one sequence directory via xml/image paths
      - many sequence directories under dataset_dir/split/<sequence>/

    Returns per item:
      {
        "obs_coords": (O,10,2),
        "obs_vis":    (O,10),
        "obs_imgs":   (O,C,H,W) or (O,H,W,C) depending on transform,
        "fut_coords": (P,10,2),
        "fut_vis":    (P,10),
        "fut_frame_idx": (P,),
        "obs_frame_idx": (O,)
      }

    Notes:
    - image path is matched by frame number
    - sequence boundaries are kept isolated
    - missing image or feature file raises FileNotFoundError
    - coordinates can optionally be normalized to [0,1]
    """

    def __init__(
        self,
        xml_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        obs_len: int = 10,
        pred_len: int = 5,
        image_transform=None,
        normalize_coords: bool = False,
        include_images: bool = True,
        feature_dir: Optional[str] = None,
        include_features: bool = False,
        image_name_format: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        split: str = "train",
    ):
        self.xml_path = xml_path
        self.image_dir = image_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.image_transform = image_transform
        self.normalize_coords = normalize_coords
        self.include_images = include_images
        self.feature_dir = feature_dir
        self.include_features = include_features
        self.image_name_format = image_name_format
        self.dataset_dir = dataset_dir
        self.split = split

        sequence_sources = self._resolve_sequence_sources()
        self.sequences = []
        self.samples = []

        for sequence_idx, source in enumerate(sequence_sources):
            frames = load_cvat_points_xml(source["xml_path"])
            if not frames:
                continue

            coords_list = []
            vis_list = []
            meta_list = []

            for fr in frames:
                coords, vis = convert_frame_to_fixed_layout(fr)

                if normalize_coords:
                    coords[:, 0] /= float(fr["width"])
                    coords[:, 1] /= float(fr["height"])

                coords_list.append(coords)
                vis_list.append(vis)
                meta_list.append(fr)

            coords_arr = np.stack(coords_list, axis=0)
            vis_arr = np.stack(vis_list, axis=0)
            coords_arr = smooth_valid_trajectory(coords_arr, vis_arr, window=7)

            frame_records = []
            for t, fr in enumerate(meta_list):
                frame_records.append(
                    {
                        "frame_idx": fr["frame_idx"],
                        "frame_name": fr["frame_name"],
                        "coords": coords_arr[t],
                        "vis": vis_arr[t],
                        "width": fr["width"],
                        "height": fr["height"],
                        "sequence_id": source["sequence_id"],
                        "image_dir": source["image_dir"],
                        "feature_dir": source["feature_dir"],
                    }
                )

            self.sequences.append(frame_records)
            T = len(frame_records)
            for start in range(T - self.total_len + 1):
                self.samples.append((sequence_idx, start))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resolve_annotation_path(sequence_dir: str) -> str:
        for annotation_name in ("instances.xml", "annotations.xml"):
            candidate = os.path.join(sequence_dir, annotation_name)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"Could not find instances.xml or annotations.xml in {sequence_dir}"
        )

    def _resolve_sequence_sources(self) -> List[Dict[str, Optional[str]]]:
        if self.dataset_dir is not None:
            split_dir = os.path.join(self.dataset_dir, self.split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Missing split directory: {split_dir}")

            sequence_sources = []
            for entry in sorted(os.listdir(split_dir)):
                sequence_dir = os.path.join(split_dir, entry)
                if not os.path.isdir(sequence_dir):
                    continue

                try:
                    xml_path = self._resolve_annotation_path(sequence_dir)
                except FileNotFoundError:
                    continue

                sequence_sources.append(
                    {
                        "sequence_id": entry,
                        "xml_path": xml_path,
                        "image_dir": sequence_dir,
                        "feature_dir": sequence_dir,
                    }
                )

            if not sequence_sources:
                raise FileNotFoundError(
                    f"No sequence folders with annotations found in {split_dir}"
                )

            return sequence_sources

        if self.xml_path is None or self.image_dir is None:
            raise ValueError(
                "Pass either dataset_dir (optionally with split) or both xml_path and image_dir."
            )

        sequence_id = os.path.basename(os.path.normpath(self.image_dir))
        resolved_feature_dir = self.feature_dir
        if self.include_features and resolved_feature_dir is None:
            resolved_feature_dir = self.image_dir

        return [
            {
                "sequence_id": sequence_id,
                "xml_path": self.xml_path,
                "image_dir": self.image_dir,
                "feature_dir": resolved_feature_dir,
            }
        ]

    def _load_feature(self, frame_idx: int, feature_dir: Optional[str]):
        if feature_dir is None:
            raise ValueError("feature_dir is None but include_features=True")

        feat_path = os.path.join(feature_dir, f"{frame_idx:06d}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        feat = torch.load(feat_path, map_location="cpu")  # (V,)
        if feat.ndim == 2 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        return feat.float()

    def _frame_idx_to_image_path(self, frame_idx: int, image_dir: Optional[str]) -> str:
        """
        Try:
        1) custom format if provided
        2) 000024.png
        3) frame_000024.png
        4) any file in folder containing frame_000024
        """
        if image_dir is None:
            raise ValueError("image_dir is None but include_images=True")

        if self.image_name_format is not None:
            candidate = os.path.join(
                image_dir, self.image_name_format.format(frame_idx)
            )
            if os.path.exists(candidate):
                return candidate

        candidate = os.path.join(image_dir, f"{frame_idx:06d}.png")
        if os.path.exists(candidate):
            return candidate

        candidate = os.path.join(image_dir, f"frame_{frame_idx:06d}.png")
        if os.path.exists(candidate):
            return candidate

        patt = re.compile(rf"frame_{frame_idx:06d}\.")
        for fn in os.listdir(image_dir):
            if patt.search(fn):
                return os.path.join(image_dir, fn)

        numeric_patt = re.compile(rf"(?:^|[^\d]){frame_idx:06d}\.(png|jpg|jpeg)$")
        for fn in os.listdir(image_dir):
            if numeric_patt.search(fn):
                return os.path.join(image_dir, fn)

        raise FileNotFoundError(
            f"Could not find image for frame {frame_idx} in {image_dir}"
        )

    def _load_image(self, frame_idx: int, image_dir: Optional[str]):
        img_path = self._frame_idx_to_image_path(frame_idx, image_dir)
        img = Image.open(img_path).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(img)
        else:
            img = np.array(img, dtype=np.uint8)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C,H,W)

        return img

    def __getitem__(self, idx: int):
        sequence_idx, start = self.samples[idx]
        frame_records = self.sequences[sequence_idx]
        chunk = frame_records[start : start + self.total_len]

        frame_indices = np.array([fr["frame_idx"] for fr in chunk], dtype=np.int64)
        coords = np.stack([fr["coords"] for fr in chunk], axis=0)  # (O+P,10,2)
        vis = np.stack([fr["vis"] for fr in chunk], axis=0)  # (O+P,10)

        obs_coords = coords[: self.obs_len].copy()
        obs_vis = vis[: self.obs_len].copy()
        fut_coords = coords[self.obs_len :].copy()
        fut_vis = vis[self.obs_len :].copy()

        # fill only observed sequence
        obs_coords_filled = forward_fill_observation_window(obs_coords, obs_vis)

        out = {
            "obs_coords": torch.tensor(
                obs_coords_filled, dtype=torch.float32
            ),  # (O,10,2)
            "obs_vis": torch.tensor(obs_vis, dtype=torch.float32),  # (O,10)
            "fut_coords": torch.tensor(fut_coords, dtype=torch.float32),  # (P,10,2)
            "fut_vis": torch.tensor(fut_vis, dtype=torch.float32),  # (P,10)
            "obs_frame_idx": torch.tensor(
                frame_indices[: self.obs_len], dtype=torch.long
            ),
            "fut_frame_idx": torch.tensor(
                frame_indices[self.obs_len :], dtype=torch.long
            ),
        }

        if self.include_features:
            obs_feats = [
                self._load_feature(fr["frame_idx"], fr["feature_dir"])
                for fr in chunk[: self.obs_len]
            ]
            fut_feats = [
                self._load_feature(fr["frame_idx"], fr["feature_dir"])
                for fr in chunk[self.obs_len :]
            ]

            out["obs_feats"] = torch.stack(obs_feats, dim=0)  # (O,V)
            out["fut_feats"] = torch.stack(fut_feats, dim=0)  # (P,V)

        if self.include_images:
            obs_imgs = [
                self._load_image(fr["frame_idx"], fr["image_dir"])
                for fr in chunk[: self.obs_len]
            ]
            out["obs_imgs"] = torch.stack(obs_imgs, dim=0)  # (O,C,H,W)

        return out


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the SurgManip dataloader.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=SURGMANIP_DIR,
        help="Dataset root containing train/ and val/ sequence folders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to read when using --dataset-dir.",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="Single-sequence annotation XML path. Overrides --dataset-dir mode.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Single-sequence image directory. Overrides --dataset-dir mode.",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Feature directory for single-sequence mode. Defaults to image_dir.",
    )
    parser.add_argument("--obs-len", type=int, default=10, help="Observation length.")
    parser.add_argument("--pred-len", type=int, default=5, help="Prediction length.")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for the smoke test."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Load observation images in the batch.",
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Load DINO features in the batch.",
    )
    parser.add_argument(
        "--normalize-coords",
        action="store_true",
        help="Normalize coordinates into [0, 1].",
    )
    return parser


def _print_tensor_summary(name: str, value) -> None:
    if isinstance(value, torch.Tensor):
        print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")
    else:
        print(f"{name}: type={type(value).__name__}")


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    dataset_kwargs = dict(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        image_transform=None,
        normalize_coords=args.normalize_coords,
        include_images=args.include_images,
        feature_dir=args.feature_dir,
        include_features=args.include_features,
    )

    if args.xml_path is not None or args.image_dir is not None:
        if args.xml_path is None or args.image_dir is None:
            raise ValueError(
                "Single-sequence mode requires both --xml-path and --image-dir."
            )
        dataset_kwargs.update(
            {
                "xml_path": args.xml_path,
                "image_dir": args.image_dir,
            }
        )
    else:
        if args.dataset_dir is None:
            raise ValueError(
                "Pass --dataset-dir, or provide both --xml-path and --image-dir."
            )
        dataset_kwargs.update(
            {
                "dataset_dir": args.dataset_dir,
                "split": args.split,
            }
        )

    ds = SurgToolSequenceDataset(**dataset_kwargs)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"dataset_size: {len(ds)}")
    print(f"batch_size: {args.batch_size}")
    print(
        "mode:",
        (
            "single-sequence"
            if "xml_path" in dataset_kwargs
            else f"dataset split '{args.split}'"
        ),
    )

    batch = next(iter(dl))
    for key, value in batch.items():
        _print_tensor_summary(key, value)
        # for i in range(value.shape[0]):
        #     print(value[i])
