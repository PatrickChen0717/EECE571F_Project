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


def temporal_moving_average(x, window_size=5):
    if window_size <= 1:
        return x

    T, M, K, C = x.shape
    y = x.copy()
    half = window_size // 2

    for m in range(M):
        for k in range(K):
            for c in range(C):
                seq = x[:, m, k, c]
                out = np.empty(T, dtype=np.float32)

                for t in range(T):
                    s = max(0, t - half)
                    e = min(T, t + half + 1)
                    vals = seq[s:e]
                    valid = vals[~np.isnan(vals)]
                    out[t] = np.mean(valid) if len(valid) > 0 else np.nan

                y[:, m, k, c] = out

    return y


class KeypointDataset(Dataset):
    def __init__(self, yaml_paths, normalize=False, smoothing=False, smoothing_window=5):
        self.yaml_paths = list(yaml_paths)
        self.groups = [[1, 2, 3, 4, 5], [8, 9, 10, 11, 12]]
        self.normalize = normalize
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window

        self.samples = []
        all_xy = []

        for path in self.yaml_paths:
            x, _ = load_yaml_episode(path, self.groups)

            if self.smoothing:
                x = temporal_moving_average(x, self.smoothing_window)

            x = x.astype(np.float32)
            self.samples.append({
                "x": torch.from_numpy(x).float(),
                "length": int(x.shape[0]),
                "episode_path": path,
            })

            if self.normalize:
                all_xy.append(x.reshape(-1, 2))

        self.mean = None
        self.std = None
        if self.normalize:
            all_xy = np.concatenate(all_xy, axis=0)
            self.mean = np.nanmean(all_xy, axis=0).astype(np.float32)
            self.std = (np.nanstd(all_xy, axis=0) + 1e-6).astype(np.float32)

            mean_t = torch.from_numpy(self.mean).float()
            std_t = torch.from_numpy(self.std).float()
            for s in self.samples:
                s["x"] = (s["x"] - mean_t) / std_t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class WindowedKeypointDataset(Dataset):
    def __init__(self, base_subset, O=10, P=5, random_window=True, img_transform=None,
                 load_from_image=True, stride=1, cache_features=True):
        self.base = base_subset
        self.O = O
        self.P = P
        self.L = O + P
        self.random_window = random_window
        self.load_from_image = load_from_image
        self.stride = stride
        self.cache_features = cache_features

        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._vr_cache = {}
        self._feature_cache = {}

        self.window_index = []
        for base_idx in range(len(self.base)):
            sample = self.base[base_idx]
            x = sample["x"]
            T = x.shape[0]

            if T < self.L:
                continue

            if self.random_window:
                self.window_index.append((base_idx, None))
            else:
                for start in range(0, T - self.L + 1, self.stride):
                    self.window_index.append((base_idx, start))

    def __len__(self):
        return len(self.window_index)

    def _get_video_path(self, episode_path):
        parts = episode_path.split(os.sep)
        video_path = os.sep.join(parts[:-1] + ["regular", "left_video.mp4"])
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        return video_path

    def _get_feature_dir(self, episode_path):
        parts = episode_path.split(os.sep)
        feat_dir = os.sep.join(parts[:-1] + ["regular", "left_frame_pt"])
        if not os.path.exists(feat_dir):
            raise FileNotFoundError(feat_dir)
        return feat_dir

    def _get_video_reader(self, episode_path):
        video_path = self._get_video_path(episode_path)
        if video_path not in self._vr_cache:
            self._vr_cache[video_path] = VideoReader(video_path, ctx=cpu(0))
        return self._vr_cache[video_path]

    def _load_frame_tensor(self, episode_path, frame_idx):
        vr = self._get_video_reader(episode_path)
        if frame_idx < 0 or frame_idx >= len(vr):
            raise IndexError(f"frame_idx={frame_idx} out of range for video {self._get_video_path(episode_path)}")
        frame_np = vr[frame_idx].asnumpy()
        img = Image.fromarray(frame_np)
        return self.img_transform(img)

    def _load_frame_sequence(self, episode_path, start_idx, length):
        vr = self._get_video_reader(episode_path)
        indices = list(range(start_idx, start_idx + length))
        if max(indices) >= len(vr):
            raise IndexError(f"Requested frames {indices[-1]} but video length is {len(vr)}")
        batch = vr.get_batch(indices).asnumpy()
        frames = [self.img_transform(Image.fromarray(arr)) for arr in batch]
        return torch.stack(frames, dim=0)

    def _load_all_features_for_episode(self, episode_path):
        if self.cache_features and episode_path in self._feature_cache:
            return self._feature_cache[episode_path]

        feat_dir = self._get_feature_dir(episode_path)
        feat_files = sorted(
            [f for f in os.listdir(feat_dir) if f.endswith(".pt")],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        if not feat_files:
            raise RuntimeError(f"No .pt feature files found in {feat_dir}")

        feats = []
        for fn in feat_files:
            feat_path = os.path.join(feat_dir, fn)
            feat = torch.load(feat_path, map_location="cpu").float()
            feats.append(feat)

        feat_tensor = torch.stack(feats, dim=0)  # (T, D)

        if self.cache_features:
            self._feature_cache[episode_path] = feat_tensor
        return feat_tensor

    def __getitem__(self, idx):
        base_idx, fixed_start = self.window_index[idx]
        sample = self.base[base_idx]

        x = sample["x"]
        episode_path = sample["episode_path"]
        L = x.shape[0]

        if L < self.L:
            raise RuntimeError(f"Sequence too short: {L} < {self.L}")

        if fixed_start is None:
            start = np.random.randint(0, L - self.L + 1)
        else:
            start = fixed_start

        obs = x[start:start + self.O]
        fut = x[start + self.O:start + self.O + self.P]

        if self.load_from_image:
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
                "obs": obs.float(),
                "fut": fut.float(),
                "obs_frames": obs_frames.float(),
                "full_frames": full_frames.float(),
                "frame": last_obs_frame.float(),
                "episode_path": episode_path,
                "start_idx": start,
            }
        else:
            all_feats = self._load_all_features_for_episode(episode_path)

            obs_feats = all_feats[start + 1 : start + self.O]              # (O-1, D)
            full_feats = all_feats[start + 1 : start + self.O + self.P]   # (O+P-1, D)
            last_obs_feat = all_feats[start + self.O - 1]                 # (D,)

            return {
                "obs": obs.float(),
                "fut": fut.float(),
                "obs_frames": obs_feats.float(),
                "full_frames": full_feats.float(),
                "frame": last_obs_feat.float(),
                "episode_path": episode_path,
                "start_idx": start,
            }