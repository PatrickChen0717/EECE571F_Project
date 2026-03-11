import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import yaml, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def load_yaml_episode(yaml_path: str, groups):
    """
    groups: list[list[int]] e.g. [[1,2,3,4,5], [8,9,10,11,12]]
    returns:
      kpts: (T, M, K, 2)
      groups: same groups
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    frame_keys = sorted(data.keys(), key=lambda k: int(k))
    T = len(frame_keys)
    M = len(groups)
    K = len(groups[0])
    assert all(len(g) == K for g in groups), "All groups must have same K"

    kpts = np.full((T, M, K, 2), np.nan, dtype=np.float32)

    for t, fk in enumerate(frame_keys):
        frame_dict = data[fk]
        for m, g in enumerate(groups):
            for j, kid in enumerate(g):
                kp = frame_dict.get(kid, frame_dict.get(str(kid), None))
                if kp is None:
                    continue
                kpts[t, m, j, 0] = float(kp[0])
                kpts[t, m, j, 1] = float(kp[1])

    return kpts, groups

class KeypointDataset(Dataset):
    def __init__(self, yaml_paths, normalize=False):
        self.yaml_paths = list(yaml_paths)
        self.groups = [[1,2,3,4,5], [8,9,10,11,12]]  # default PSM1 + PSM3
        self.normalize = normalize

        self.mean = None
        self.std = None
        if normalize:
            all_xy = []
            for p in self.yaml_paths:
                kpts, _ = load_yaml_episode(p, self.groups)   # (T,M,K,2)
                all_xy.append(kpts.reshape(-1, 2))
            all_xy = np.concatenate(all_xy, axis=0)
            self.mean = np.nanmean(all_xy, axis=0).astype(np.float32)
            self.std = (np.nanstd(all_xy, axis=0) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.yaml_paths)

    def __getitem__(self, idx):
        path = self.yaml_paths[idx]
        x, groups = load_yaml_episode(path, self.groups)  # (T,M,5,2)

        if self.normalize:
            x = (x - self.mean) / self.std

        return {
            "x": torch.from_numpy(x).float(),  # (T,M,5,2)
            "length": torch.tensor(x.shape[0]),
            "episode_path": path,
        }
        

class WindowedKeypointDataset(Dataset):
    def __init__(self, base_subset, O=10, P=5, random_window=True, img_transform=None):
        self.base = base_subset
        self.O = O
        self.P = P
        self.L = O + P
        self.random_window = random_window
        
        
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def __len__(self):
        return len(self.base)

    def _get_frame_path(self, episode_path, frame_idx):
        """
        Modify this function to match your actual dataset structure.

        Example assumption:
        - YAML path: .../keypoints_left.yaml
        - Image path: .../left_frames/000123.png
        or       : .../images_left/000123.png
        """
        episode_dir = os.path.dirname(episode_path)

        candidate = os.path.join(episode_dir, "regular/left_frames", f"{frame_idx:06d}.png")
        if os.path.exists(candidate):
            return candidate

        raise FileNotFoundError(
            f"Could not find frame for episode_path={episode_path}, frame_idx={frame_idx}"
        )
        
    def __getitem__(self, idx):
        item = self.base[idx]
        x = item["x"]  # (T,M,5,2)
        episode_path = item["episode_path"]

        T_full = x.shape[0]
        if T_full < self.L:
            raise RuntimeError(f"Sequence too short: {T_full} < {self.L}")

        start = torch.randint(0, T_full - self.L + 1, (1,)).item() if self.random_window else 0

        chunk = x[start:start + self.L]   # (O+P, M, 5, 2)
        obs = chunk[:self.O]              # (O, M, 5, 2)
        fut = chunk[self.O:]              # (P, M, 5, 2)

        frame_idx = start + self.O - 1
        frame_path = self._get_frame_path(episode_path, frame_idx)

        img = Image.open(frame_path).convert("RGB")
        frame_tensor = self.img_transform(img)   # (3,224,224)
        
        return {"obs": obs, "fut": fut,  "frame": frame_tensor}