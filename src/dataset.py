from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DeepfakeTwoStreamDataset(Dataset):
    def __init__(self, csv_path: str, split: str, augment: bool = False):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.augment = augment

        # ImageNet normalization (for RGB only)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.df)

    def _norm(self, x):
        return (x - self.mean) / self.std

    def _augment_rgb(self, x):
        # x shape: [T, C, H, W], range [0, 1]

        if np.random.rand() < 0.5:
            # Horizontal flip
            x = torch.flip(x, dims=[3])

        if np.random.rand() < 0.3:
            # Rotation (ONLY for RGB)
            x = torch.rot90(x, k=int(np.random.choice([1, 2, 3])), dims=[2, 3])

        if np.random.rand() < 0.5:
            # Color jitter
            brightness = 1.0 + np.random.uniform(-0.15, 0.15)
            contrast = 1.0 + np.random.uniform(-0.20, 0.20)
            x = x * brightness
            x_mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - x_mean) * contrast + x_mean

        if np.random.rand() < 0.4:
            # Gaussian noise
            noise_std = float(np.random.uniform(0.0, 0.03))
            x = x + torch.randn_like(x) * noise_std

        if np.random.rand() < 0.3:
            # Blur
            k = 3 if np.random.rand() < 0.7 else 5
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

        if np.random.rand() < 0.3:
            # Compression simulation
            levels = float(np.random.choice([32, 64, 128]))
            x = torch.round(x * levels) / levels

        return x.clamp(0.0, 1.0)

    def _augment_flow(self, flow):
        # flow shape: [T, 2, H, W] where channels are (dx, dy)
        # Keep flow augmentation minimal and geometry-safe.

        if np.random.rand() < 0.5:
            flow = torch.flip(flow, dims=[3])
            # Horizontal flip changes sign of horizontal motion.
            flow[:, 0] = -flow[:, 0]

        return flow

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz = np.load(Path(row["npz_path"]))

        # -------- RGB --------
        rgb = torch.from_numpy(npz["rgb"]).permute(0, 3, 1, 2).float() / 255.0

        # -------- FLOW (2-channel normalized optical flow) --------
        flow = torch.from_numpy(npz["flow"]).permute(0, 3, 1, 2).float()

        # -------- AUGMENTATION --------
        if self.augment:
            rgb = self._augment_rgb(rgb)
            flow = self._augment_flow(flow)

        # -------- NORMALIZE RGB --------
        rgb = self._norm(rgb)

        label = torch.tensor(int(npz["label"]), dtype=torch.long)

        return rgb, flow, label