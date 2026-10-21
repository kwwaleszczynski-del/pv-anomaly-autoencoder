"""Evaluation against the injected faults: point-wise and event-level metrics.

Point labels: a daylight 15-minute step is positive when a fault is active and changes the plant output (or the
irradiance reading) by at least IMPACT_MIN. Steps where a fault is active but has a smaller effect (soiling that has
just started, shading when the sun is elsewhere, a derated inverter on a cloudy day) are left out of the point
metrics instead of being counted as normal.

Per-type precision is computed on healthy points plus points of that type, so all types share the same false
positives; the 'all' row pools every type.

Usage:
    from pv_anomaly.evaluate import ground_truth, point_metrics, event_metrics
    truth = ground_truth(df, feats["daylight"])
    pts = point_metrics("autoencoder", alarm, score, truth, test_mask)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_anomaly.simulate import FAULT_TYPES

IMPACT_MIN = 0.03


def ground_truth(df: pd.DataFrame, daylight) -> pd.DataFrame:
    active = (df["fault"] != "").to_numpy()
    label = active & (df["impact"] >= IMPACT_MIN).to_numpy()
    daylight = np.asarray(daylight, bool)
    return pd.DataFrame({"daylight": daylight, "active": active, "label": label,
                         "kind": df["fault"].where(label, "").to_numpy(),
                         "evaluated": daylight & (~active | label)}, index=df.index)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise), threshold-free."""
    ok = np.isfinite(score)
    s, y = score[ok], label[ok].astype(bool)
    if not y.any():
        return float("nan")
    y = y[np.argsort(-s, kind="stable")]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(precision[y].sum() / y.sum())


def pr_curve(score: np.ndarray, label: np.ndarray, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Recall and precision at every score threshold (thinned to about `points` points for plotting)."""
    ok = np.isfinite(score)
    y = label[ok].astype(bool)[np.argsort(-score[ok], kind="stable")]
    tp = np.cumsum(y)
    recall, precision = tp / max(y.sum(), 1), tp / np.arange(1, len(y) + 1)
    step = max(1, len(y) // points)
    return recall[::step], precision[::step]


def point_metrics(method: str, alarm: pd.Series, score: pd.Series, truth: pd.DataFrame, mask) -> list[dict]:
    m = np.asarray(mask, bool) & truth["evaluated"].to_numpy()
    a, s = alarm.to_numpy(bool), score.to_numpy(float)
    label, kind = truth["label"].to_numpy(), truth["kind"].to_numpy()
    normal = m & ~label
    fp = int((a & normal).sum())
    rows = []
    for k in ("all", *FAULT_TYPES):
        pos = m & label if k == "all" else m & (kind == k)
        tp, fn = int((a & pos).sum()), int((~a & pos).sum())
        p, r, f1 = _prf(tp, fp, fn)
        sel = normal | pos
        rows.append({"method": method, "fault": k, "precision": p, "recall": r, "f1": f1,
                     "avg_precision": average_precision(s[sel], pos[sel]), "positives": int(pos.sum())})
    return rows


def event_metrics(method: str, alarm: pd.Series, events: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    """Per injected event: detected (an alarm while the fault has a labelled effect - alarms at the very start of
    a gradual fault are more likely coincidence than detection) and time to detection from the event start."""
    rows = []
    hit = alarm.to_numpy(bool) & truth["label"].to_numpy()
    for ev in events.itertuples():
        inside = (alarm.index >= ev.start) & (alarm.index < ev.end)
        hits = alarm.index[inside & hit]
        ttd = (hits[0] - ev.start) / pd.Timedelta("1h") if len(hits) else np.nan
        rows.append({"method": method, "fault_id": ev.fault_id, "fault": ev.kind, "severity": ev.severity,
                     "detected": bool(len(hits)), "ttd_hours": ttd})
    return rows


def false_alarms(method: str, alarm: pd.Series, truth: pd.DataFrame, mask) -> dict:
    """Alarm episodes (runs of consecutive alarmed daylight steps) that start while no fault is active."""
    m = np.asarray(mask, bool) & truth["daylight"].to_numpy()
    seq = alarm.to_numpy(bool)[m]
    starts = seq & ~np.r_[False, seq[:-1]]
    healthy = ~truth["active"].to_numpy()[m]
    healthy_days = (np.asarray(mask, bool) & ~truth["active"].to_numpy()).sum() * 0.25 / 24
    n = int((starts & healthy).sum())
    return {"method": method, "false_alarm_episodes": n, "healthy_days": round(float(healthy_days), 1),
            "per_30_days": n / healthy_days * 30 if healthy_days else np.nan}


def summarize_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Detection rate and median / max time to detection per method and fault type."""
    g = ev.groupby(["method", "fault"], sort=False)
    return pd.DataFrame({"events": g.size(), "detected": g["detected"].sum(), "detection_rate": g["detected"].mean(),
                         "median_ttd_hours": g["ttd_hours"].median(),
                         "max_ttd_hours": g["ttd_hours"].max()}).reset_index()
