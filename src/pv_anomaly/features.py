"""Feature engineering, train/validation/test split and sliding windows.

Features per 15-minute step, all computable from plant measurements and the timestamp:

    p_norm                       AC power / inverter rating
    pr                           AC power / (DC nameplate * POA / 1000): power normalised to irradiance,
                                 set to NOMINAL_PR below LOW_POA where the ratio is meaningless
    poa                          plane-of-array irradiance / 1000
    kc                           POA / clear-sky POA: exposes irradiance-sensor problems on clear days
    t_mod, t_amb                 module and ambient temperature [degC]
    sun_el, sun_az_sin, sun_az_cos  sun position: time of day and season

Usage:
    from pv_anomaly.features import FEATURES, build_features, make_windows
    feats = build_features(df, plant)
    X, end_pos = make_windows(feats[list(FEATURES)].to_numpy(), window=8)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_anomaly.plant import PlantConfig, solar_geometry

FEATURES = ("p_norm", "pr", "poa", "kc", "t_mod", "t_amb", "sun_el", "sun_az_sin", "sun_az_cos")
MIN_ELEVATION = 10.0  # points with the sun lower than this are neither trained on nor evaluated
LOW_POA = 50.0  # W/m2; below this the irradiance ratios are undefined
NOMINAL_PR = 0.8  # placeholder for an undefined performance ratio (zeros would dominate the feature's spread)


def build_features(df: pd.DataFrame, plant: PlantConfig) -> pd.DataFrame:
    """Feature table on the index of `df` (needs p_ac, poa, t_mod, t_amb) plus a boolean ``daylight`` column."""
    geo = solar_geometry(df.index, plant)
    poa = df["poa"]
    pr = (df["p_ac"] / (plant.pdc0_kw * poa / 1000)).clip(0, 1.5).mask(poa < LOW_POA, NOMINAL_PR)
    kc = (poa / geo["poa_clearsky"]).mask(geo["poa_clearsky"] < LOW_POA, 0.0).clip(0, 2)
    az = np.radians(geo["azimuth"])
    return pd.DataFrame({
        "p_norm": df["p_ac"] / plant.pac0_kw, "pr": pr, "poa": poa / 1000, "kc": kc,
        "t_mod": df["t_mod"], "t_amb": df["t_amb"],
        "sun_el": geo["elevation"].clip(lower=-10) / 90, "sun_az_sin": np.sin(az), "sun_az_cos": np.cos(az),
        "daylight": geo["elevation"] >= MIN_ELEVATION,
    }, index=df.index)


def split_masks(index: pd.DatetimeIndex, tz: str, train_end: str, val_frac: float = 0.2,
                seed: int = 0, train_start: str | None = None) -> dict[str, np.ndarray]:
    """Boolean masks ``train``, ``val`` (random whole local days of [`train_start`, `train_end`)) and ``test``
    (all other days)."""
    day = index.tz_convert(tz).normalize().tz_localize(None)
    before = (day < pd.Timestamp(train_end)) & (day >= pd.Timestamp(train_start or day.min()))
    days = np.unique(day[before])
    rng = np.random.default_rng(seed)
    val_days = rng.choice(days, size=int(round(val_frac * len(days))), replace=False)
    val = np.asarray(before & day.isin(val_days))
    return {"train": np.asarray(before) & ~val, "val": val, "test": ~np.asarray(before)}


def make_windows(X: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened windows of `window` consecutive rows (time-major) and the row position of each window's end.

    Rows must be on a regular time grid; windows containing NaN are dropped.
    """
    v = np.lib.stride_tricks.sliding_window_view(np.asarray(X, np.float32), window, axis=0)  # (n-w+1, F, w)
    v = v.transpose(0, 2, 1).reshape(v.shape[0], -1)
    end = np.arange(window - 1, len(X))
    ok = ~np.isnan(v).any(axis=1)
    return v[ok], end[ok]
