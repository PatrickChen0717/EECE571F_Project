import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu


def load_yaml_episode(yaml_path: str, groups):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    frame_keys = sorted(data.keys(), key=lambda k: int(k))
    T = len(frame_keys)
    M = len(groups)
    K = len(groups[0])
    assert all(len(g) == K for g in groups)

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
        self.groups = [[1, 2, 3, 4, 5], [8, 9, 10, 11, 12]]
        self.normalize = normalize

        self.mean = None
        self.std = None
        if normalize:
            all_xy = []
            for p in self.yaml_paths:
                kpts, _ = load_yaml_episode(p, self.groups)
                all_xy.append(kpts.reshape(-1, 2))
            all_xy = np.concatenate(all_xy, axis=0)
            self.mean = np.nanmean(all_xy, axis=0).astype(np.float32)
            self.std = (np.nanstd(all_xy, axis=0) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.yaml_paths)

    def __getitem__(self, idx):
        path = self.yaml_paths[idx]
        x, _ = load_yaml_episode(path, self.groups)

        if self.normalize:
            x = (x - self.mean) / self.std

        return {
            "x": torch.from_numpy(x).float(),
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

        self._vr_cache = {}

    def __len__(self):
        return len(self.base)

    def _get_video_path(self, episode_path):
        # split path
        parts = episode_path.split(os.sep)

        # remove filename and append video path
        parts = parts[:-1] + ["regular", "left_video.mp4"]

        video_path = os.sep.join(parts)

        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        return video_path

    def _get_video_reader(self, episode_path):
        video_path = self._get_video_path(episode_path)
        if video_path not in self._vr_cache:
            self._vr_cache[video_path] = VideoReader(video_path, ctx=cpu(0))
        return self._vr_cache[video_path]

    def _load_frame_tensor(self, episode_path, frame_idx):
        vr = self._get_video_reader(episode_path)

        if frame_idx < 0 or frame_idx >= len(vr):
            raise IndexError(f"frame_idx={frame_idx} out of range for video {self._get_video_path(episode_path)}")

        frame_np = vr[frame_idx].asnumpy()   # HWC RGB uint8
        img = Image.fromarray(frame_np)
        return self.img_transform(img)

    def _load_frame_sequence(self, episode_path, start_idx, length):
        vr = self._get_video_reader(episode_path)
        indices = list(range(start_idx, start_idx + length))

        if max(indices) >= len(vr):
            raise IndexError(f"Requested frames {indices[-1]} but video length is {len(vr)}")

        batch = vr.get_batch(indices).asnumpy()   # (T,H,W,3)
        frames = [self.img_transform(Image.fromarray(arr)) for arr in batch]
        return torch.stack(frames, dim=0)   # (T,3,224,224)

    def __getitem__(self, idx):
        sample = self.base[idx]

        x = sample["x"]
        episode_path = sample["episode_path"]
        L = x.shape[0]

        if L < self.L:
            raise RuntimeError(f"Sequence too short: {L} < {self.L}")

        if self.random_window:
            start = np.random.randint(0, L - self.L + 1)
        else:
            start = 0

        obs = x[start:start + self.O]
        fut = x[start + self.O:start + self.O + self.P]

        obs_frames = self._load_frame_sequence(
            episode_path=episode_path,
            start_idx=start + 1,
            length=self.O - 1
        )

        full_frames = self._load_frame_sequence(
            episode_path=episode_path,
            start_idx=start + 1,
            length=self.O + self.P - 1
        )

        last_obs_frame = self._load_frame_tensor(
            episode_path=episode_path,
            frame_idx=start + self.O - 1
        )

        return {
            "obs": torch.as_tensor(obs).float(),
            "fut": torch.as_tensor(fut).float(),
            "obs_frames": obs_frames.float(),
            "full_frames": full_frames.float(),
            "frame": last_obs_frame.float(),
            "episode_path": episode_path,
            "start_idx": start,
        }