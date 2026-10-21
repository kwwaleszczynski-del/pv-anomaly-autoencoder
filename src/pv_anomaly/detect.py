"""From raw scores to alarms: causal smoothing over daylight points and a threshold set on healthy data.

The same post-processing is applied to every detector so that they are compared at the same false-alarm budget:
the threshold is the `q` quantile of the smoothed score on validation days (healthy, never used for training).

Usage:
    from pv_anomaly.detect import smooth, calibrate
    s = smooth(raw_score, daylight, k=8, signed=True)
    alarm = s > calibrate(s, val_mask, q=0.995)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smooth(raw: pd.Series, daylight, k: int, signed: bool) -> pd.Series:
    """Mean of the last `k` daylight scores (NaN outside daylight).

    For signed residuals the absolute value of the mean is taken, so consistent offsets in either direction
    add up while noise cancels; reconstruction errors are already non-negative.
    """
    m = np.asarray(daylight, bool)
    s = raw[m].rolling(k, min_periods=max(1, k // 2)).mean()
    return (s.abs() if signed else s).reindex(raw.index)


def calibrate(score: pd.Series, mask, q: float) -> float:
    """Threshold = `q` quantile of `score` over `mask` (healthy validation points)."""
    return float(np.nanquantile(score[np.asarray(mask, bool)], q))
