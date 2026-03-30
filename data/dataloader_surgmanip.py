import os
import re
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


LABEL_ORDER = ["shaft", "wrist", "ee", "tip1", "tip2"]
NUM_TOOL_KPTS = 5
NUM_TOOLS = 2
NUM_KEYPOINTS = NUM_TOOL_KPTS * NUM_TOOLS  # 10


def parse_frame_number(name: str) -> int:
    """
    Extract integer frame number from strings like:
      frame_000024
      surgmanip_cn_suturing_30hz_frame_000024.png
    """
    m = re.search(r"frame_(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse frame index from name: {name}")
    return int(m.group(1))


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

            points_by_label[label].append({
                "xy": xy,
                "occluded": occluded,
            })

        frames.append({
            "frame_idx": frame_idx,
            "width": width,
            "height": height,
            "points_by_label": points_by_label,
        })

    frames.sort(key=lambda x: x["frame_idx"])
    return frames


def assign_two_points_to_left_right(points: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    If there are 2 visible points for a label, assign smaller x to left tool, larger x to right tool.
    If there is 1 point, left/right ambiguity remains unresolved here.
    """
    if len(points) == 0:
        return None, None
    if len(points) == 1:
        return points[0], None  # temporary; caller may reassign using tracking
    pts_sorted = sorted(points, key=lambda p: p[0])
    return pts_sorted[0], pts_sorted[1]


def build_tool_centers_from_partial(frame_coords: np.ndarray, frame_vis: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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

    left_center = np.mean(left_pts, axis=0).astype(np.float32) if len(left_pts) > 0 else None
    right_center = np.mean(right_pts, axis=0).astype(np.float32) if len(right_pts) > 0 else None
    return left_center, right_center


def assign_single_point_with_tracking(
    pt: np.ndarray,
    prev_left_center: Optional[np.ndarray],
    prev_right_center: Optional[np.ndarray]
) -> str:
    """
    Assign a single ambiguous point to left/right using previous frame tool centers.
    Fallback: x < mid => left else right.
    """
    if prev_left_center is None and prev_right_center is None:
        return "left" if pt[0] < 320 else "right"
    if prev_left_center is None:
        return "right"
    if prev_right_center is None:
        return "left"

    d_left = point_distance(pt, prev_left_center)
    d_right = point_distance(pt, prev_right_center)
    return "left" if d_left <= d_right else "right"


def convert_frame_to_fixed_layout(frame_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one frame into fixed 10-keypoint layout.

    Order:
      left:  shaft,wrist,ee,tip1,tip2
      right: shaft,wrist,ee,tip1,tip2

    Rule:
    - if 2+ points for a label: smaller x -> left, larger x -> right
    - if 1 point: x < width/2 -> left, else right
    - occluded=1 => keep coord but vis=0
    """
    coords = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    vis = np.zeros((NUM_KEYPOINTS,), dtype=np.float32)

    img_mid_x = frame_dict["width"] / 2.0

    for label_idx, label in enumerate(LABEL_ORDER):
        items = frame_dict["points_by_label"][label]

        pts = []
        for item in items:
            pts.append((item["xy"], item["occluded"]))

        # sort by x from left to right
        pts = sorted(pts, key=lambda t: t[0][0])

        if len(pts) >= 2:
            left_pt, left_occ = pts[0]
            right_pt, right_occ = pts[-1]

            li = label_idx
            ri = NUM_TOOL_KPTS + label_idx

            coords[li] = left_pt
            vis[li] = 0.0 if left_occ == 1 else 1.0

            coords[ri] = right_pt
            vis[ri] = 0.0 if right_occ == 1 else 1.0

        elif len(pts) == 1:
            pt, occ = pts[0]

            if pt[0] < img_mid_x:
                idx = label_idx
            else:
                idx = NUM_TOOL_KPTS + label_idx

            coords[idx] = pt
            vis[idx] = 0.0 if occ == 1 else 1.0

    return coords, vis


def forward_fill_observation_window(obs_coords: np.ndarray, obs_vis: np.ndarray) -> np.ndarray:
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

import numpy as np

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
                    seg = coords[start:end, n]   # (L,2)
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
    Dataset for:
      - one XML annotation file
      - one left_frames directory

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
    - missing image file raises FileNotFoundError
    - coordinates can optionally be normalized to [0,1]
    """

    def __init__(
        self,
        xml_path: str,
        image_dir: str,
        obs_len: int = 10,
        pred_len: int = 5,
        image_transform=None,
        normalize_coords: bool = False,
        include_images: bool = True,
        feature_dir: str = None,         
        include_features: bool = False,
        image_name_format: Optional[str] = None,
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

        frames = load_cvat_points_xml(xml_path)

        # Build fixed-layout per-frame tensors with simple temporal assignment
        coords_list = []
        vis_list = []
        meta_list = []

        for fr in frames:
            coords, vis = convert_frame_to_fixed_layout(fr)

            if normalize_coords:
                coords[:, 0] /= float(fr["width"])
                coords[:, 1] /= float(fr["height"])

            coords_list.append(coords)   # (10,2)
            vis_list.append(vis)         # (10,)
            meta_list.append(fr)
        
        coords_arr = np.stack(coords_list, axis=0)   # (T,10,2)
        vis_arr = np.stack(vis_list, axis=0)         # (T,10)
        coords_arr = smooth_valid_trajectory(coords_arr, vis_arr, window=7)

        self.frame_records = []

        for t, fr in enumerate(meta_list):
            self.frame_records.append({
                "frame_idx": fr["frame_idx"],
                "coords": coords_arr[t],
                "vis": vis_arr[t],
                "width": fr["width"],
                "height": fr["height"],
            })
    
        self.samples = []
        T = len(self.frame_records)
        for start in range(T - self.total_len + 1):
            self.samples.append(start)

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, frame_idx: int):
        if self.feature_dir is None:
            raise ValueError("feature_dir is None but include_features=True")

        feat_path = os.path.join(self.feature_dir, f"{frame_idx:06d}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        feat = torch.load(feat_path, map_location="cpu")   # (V,)
        if feat.ndim == 2 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        return feat.float()
    
    def _frame_idx_to_image_path(self, frame_idx: int) -> str:
        """
        Try:
        1) custom format if provided
        2) frame_000024.png
        3) any file in folder containing frame_000024
        """
        if self.image_name_format is not None:
            candidate = os.path.join(self.image_dir, self.image_name_format.format(frame_idx))
            if os.path.exists(candidate):
                return candidate

        candidate = os.path.join(self.image_dir, f"frame_{frame_idx:06d}.png")
        if os.path.exists(candidate):
            return candidate

        patt = re.compile(rf"frame_{frame_idx:06d}\.")
        for fn in os.listdir(self.image_dir):
            if patt.search(fn):
                return os.path.join(self.image_dir, fn)

        raise FileNotFoundError(f"Could not find image for frame {frame_idx} in {self.image_dir}")

    def _load_image(self, frame_idx: int):
        img_path = self._frame_idx_to_image_path(frame_idx)
        img = Image.open(img_path).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(img)
        else:
            img = np.array(img, dtype=np.uint8)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C,H,W)

        return img

    def __getitem__(self, idx: int):
        start = self.samples[idx]
        chunk = self.frame_records[start:start + self.total_len]

        frame_indices = np.array([fr["frame_idx"] for fr in chunk], dtype=np.int64)
        coords = np.stack([fr["coords"] for fr in chunk], axis=0)  # (O+P,10,2)
        vis = np.stack([fr["vis"] for fr in chunk], axis=0)        # (O+P,10)

        obs_coords = coords[:self.obs_len].copy()
        obs_vis = vis[:self.obs_len].copy()
        fut_coords = coords[self.obs_len:].copy()
        fut_vis = vis[self.obs_len:].copy()

        # fill only observed sequence
        obs_coords_filled = forward_fill_observation_window(obs_coords, obs_vis)

        out = {
            "obs_coords": torch.tensor(obs_coords_filled, dtype=torch.float32),  # (O,10,2)
            "obs_vis": torch.tensor(obs_vis, dtype=torch.float32),                # (O,10)
            "fut_coords": torch.tensor(fut_coords, dtype=torch.float32),          # (P,10,2)
            "fut_vis": torch.tensor(fut_vis, dtype=torch.float32),                # (P,10)
            "obs_frame_idx": torch.tensor(frame_indices[:self.obs_len], dtype=torch.long),
            "fut_frame_idx": torch.tensor(frame_indices[self.obs_len:], dtype=torch.long),
        }

        if self.include_features:
            obs_feats = [self._load_feature(int(fi)) for fi in frame_indices[:self.obs_len]]
            fut_feats = [self._load_feature(int(fi)) for fi in frame_indices[self.obs_len:]]

            out["obs_feats"] = torch.stack(obs_feats, dim=0)   # (O,V)
            out["fut_feats"] = torch.stack(fut_feats, dim=0)   # (P,V)

        if self.include_images:
            obs_imgs = [self._load_image(int(fi)) for fi in frame_indices[:self.obs_len]]
            out["obs_imgs"] = torch.stack(obs_imgs, dim=0)     # (O,C,H,W)

        return out