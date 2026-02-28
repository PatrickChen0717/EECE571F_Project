import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import yaml, numpy as np, torch
from torch.utils.data import Dataset


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
    def __init__(self, base_subset, O=10, P=5, random_window=True):
        self.base = base_subset
        self.O = O
        self.P = P
        self.L = O + P
        self.random_window = random_window

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x = item["x"]  # (T,M,5,2)

        T_full = x.shape[0]
        if T_full < self.L:
            raise RuntimeError(f"Sequence too short: {T_full} < {self.L}")

        start = torch.randint(0, T_full - self.L + 1, (1,)).item() if self.random_window else 0

        chunk = x[start:start + self.L]   # (O+P, M, 5, 2)
        obs = chunk[:self.O]              # (O, M, 5, 2)
        fut = chunk[self.O:]              # (P, M, 5, 2)

        return {"obs": obs, "fut": fut}