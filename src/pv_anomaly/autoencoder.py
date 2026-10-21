"""Dense autoencoder over sliding windows of features; anomaly score = reconstruction error.

Each sample is a window of `window` consecutive 15-minute steps (2 h by default) of all features, standardised
with training statistics and flattened. The network is trained on healthy data only; through the bottleneck it
learns how power, irradiance, temperature and sun position normally go together, so its reconstruction of a faulty
window is pulled back towards healthy behaviour. Two scores are offered for the window ending at each timestamp
(causal: only past data are used): the signed reconstruction residual of the performance ratio, and the classic
mean squared error over all features. Training runs on the CPU in well under a minute.

Usage:
    from pv_anomaly.autoencoder import AEConfig, AutoencoderDetector
    det = AutoencoderDetector(AEConfig(epochs=100, seed=42))
    history = det.fit(feats, train_mask, val_mask)
    score = det.score(feats)            # or det.score(feats, kind="mse")
    det.save("models/autoencoder.pt"); det = AutoencoderDetector.load("models/autoencoder.pt")
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pv_anomaly.features import FEATURES, make_windows

log = logging.getLogger(__name__)


@dataclass
class AEConfig:
    window: int = 8
    hidden: tuple[int, ...] = (64, 32)
    latent: int = 6
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    seed: int = 42


class WindowAutoencoder(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], latent: int):
        super().__init__()
        enc, d = [], n_in
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.encoder = nn.Sequential(*enc, nn.Linear(d, latent))
        dec, d = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.decoder = nn.Sequential(*dec, nn.Linear(d, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    name = "autoencoder"

    def __init__(self, config: AEConfig = AEConfig(), features: tuple[str, ...] = FEATURES):
        self.config = config
        self.features = tuple(features)
        self.mean = np.zeros(len(self.features))
        self.std = np.ones(len(self.features))
        self.extra: dict = {}
        self.model = WindowAutoencoder(config.window * len(self.features), tuple(config.hidden), config.latent)

    def _windows(self, feats: pd.DataFrame) -> tuple[torch.Tensor, np.ndarray]:
        X = (feats[list(self.features)].to_numpy(float) - self.mean) / self.std
        W, end = make_windows(X, self.config.window)
        return torch.from_numpy(W), end

    def fit(self, feats: pd.DataFrame, train_mask, val_mask) -> list[dict]:
        """Train on windows whose last step is in `train_mask` (and in daylight); early stopping on `val_mask`."""
        cfg = self.config
        torch.manual_seed(cfg.seed)
        self.model = WindowAutoencoder(cfg.window * len(self.features), tuple(cfg.hidden), cfg.latent)
        daylight = feats["daylight"].to_numpy()
        train_mask, val_mask = np.asarray(train_mask) & daylight, np.asarray(val_mask) & daylight
        ref = feats.loc[train_mask, list(self.features)]
        self.mean, self.std = ref.mean().to_numpy(), ref.std().to_numpy() + 1e-6

        W, end = self._windows(feats)
        Xtr, Xva = W[train_mask[end]], W[val_mask[end]]
        log.info("autoencoder: %d training and %d validation windows of %d values", len(Xtr), len(Xva), W.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(cfg.seed)
        best, best_state, wait, history = np.inf, None, 0, []
        for epoch in range(cfg.epochs):
            self.model.train()
            total = 0.0
            for idx in torch.randperm(len(Xtr), generator=gen).split(cfg.batch_size):
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(Xtr[idx]), Xtr[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.model.eval()
            with torch.no_grad():
                val = nn.functional.mse_loss(self.model(Xva), Xva).item()
            history.append({"epoch": epoch + 1, "train_loss": total / len(Xtr), "val_loss": val})
            if val < best - 1e-5:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        self.model.load_state_dict(best_state)
        log.info("autoencoder: %d epochs, best validation loss %.4f", len(history), best)
        return history

    def score(self, feats: pd.DataFrame, kind: str = "pr_residual") -> pd.Series:
        """Score of the window ending at each timestamp (NaN where no complete window exists).

        ``pr_residual``: measured minus reconstructed performance ratio, averaged over the window, in units of its
        training spread - signed like the baseline residuals (negative = less power than the network expects).
        ``mse``: mean squared reconstruction error over all features and steps (the textbook score).
        """
        W, end = self._windows(feats)
        self.model.eval()
        with torch.no_grad():
            R = (W - self.model(W)).numpy().reshape(len(W), self.config.window, len(self.features))
        if kind == "pr_residual":
            val = R[:, :, self.features.index("pr")].mean(axis=1)
        elif kind == "mse":
            val = (R**2).mean(axis=(1, 2))
        else:
            raise ValueError(f"unknown score kind {kind!r}")
        out = np.full(len(feats), np.nan)
        out[end] = val
        return pd.Series(out, index=feats.index, name=f"{self.name}_{kind}")

    def save(self, path: Path, **extra) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": {**asdict(self.config), "hidden": list(self.config.hidden)},
                    "features": list(self.features), "mean": torch.tensor(self.mean), "std": torch.tensor(self.std),
                    "state_dict": self.model.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderDetector":
        ckpt = torch.load(path, weights_only=True)
        cfg = AEConfig(**{**ckpt["config"], "hidden": tuple(ckpt["config"]["hidden"])})
        det = cls(cfg, tuple(ckpt["features"]))
        det.mean, det.std = ckpt["mean"].numpy(), ckpt["std"].numpy()
        det.model.load_state_dict(ckpt["state_dict"])
        det.extra = {k: v for k, v in ckpt.items() if k not in ("config", "features", "mean", "std", "state_dict")}
        return det
