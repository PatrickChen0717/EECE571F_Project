import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import yaml, numpy as np, torch
from torch.utils.data import Dataset

def load_yaml_episode(yaml_path: str, kpt_ids=None):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # frames sorted by time index
    frame_keys = sorted(data.keys(), key=lambda k: int(k))
    first_frame = data[frame_keys[0]]

    all_ids = sorted(int(k) for k in first_frame.keys())
    if kpt_ids is None:
        kpt_ids = all_ids
    elif isinstance(kpt_ids, int):
        kpt_ids = all_ids[:kpt_ids]
    else:
        kpt_ids = [int(x) for x in kpt_ids]

    T, K = len(frame_keys), len(kpt_ids)
    kpts = np.full((T, K, 2), np.nan, dtype=np.float32)

    for t, fk in enumerate(frame_keys):
        frame_dict = data[fk]
        for j, kid in enumerate(kpt_ids):
            kp = frame_dict.get(kid, frame_dict.get(str(kid), None))
            if kp is None:
                continue
            kpts[t, j, 0] = float(kp[0])
            kpts[t, j, 1] = float(kp[1])

    return kpts, kpt_ids  # (T,K,2), list[int]



class KeypointDataset(Dataset):
    """
    One YAML file = one episode sequence.

    Returns:
      x: (T, K, 2)  full sequence
      y: (T, K, 2)  next-step target (shifted), last frame copied (or you can drop it)
      length: T
    """
    def __init__(self, yaml_paths, kpt_ids=5, normalize=False):
        self.yaml_paths = list(yaml_paths)
        self.kpt_ids = kpt_ids
        self.normalize = normalize

        # Optional global normalization: load all once to compute stats
        self.mean = None
        self.std = None
        if normalize:
            all_xy = []
            for p in self.yaml_paths:
                kpts, _ = load_yaml_episode(p, kpt_ids=kpt_ids)
                all_xy.append(kpts.reshape(-1, 2))
            all_xy = np.concatenate(all_xy, axis=0)
            self.mean = np.nanmean(all_xy, axis=0).astype(np.float32)
            self.std = (np.nanstd(all_xy, axis=0) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.yaml_paths)

    def __getitem__(self, idx):
        path = self.yaml_paths[idx]
        x, ids = load_yaml_episode(path, kpt_ids=self.kpt_ids)  # (T,K,2)

        if self.normalize:
            x = (x - self.mean) / self.std

        # next-step target
        y = np.empty_like(x)
        y[:-1] = x[1:]
        y[-1] = x[-1]

        return {
            "x": torch.from_numpy(x).float(),      # (T,K,2)
            "y": torch.from_numpy(y).float(),      # (T,K,2)
            "length": torch.tensor(x.shape[0]),
            "episode_path": path,
            "kpt_ids": torch.tensor(ids, dtype=torch.long),
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
        x = item["x"]

        T_full = x.shape[0]
        if T_full < self.L:
            raise RuntimeError(f"Sequence too short: {T_full} < {self.L}")

        if self.random_window:
            start = torch.randint(0, T_full - self.L + 1, (1,)).item()
        else:
            start = 0

        chunk = x[start:start + self.L]   # (O+P, K, 2)
        obs = chunk[:self.O]              # (O, K, 2)
        fut = chunk[self.O:]              # (P, K, 2)

        return {"obs": obs, "fut": fut}