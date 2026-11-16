"""Evaluation against the injected faults: point-wise and event-level metrics.

Point labels: a daylight 15-minute step is positive when a fault is active and changes the plant output (or the
irradiance reading) by at least IMPACT_MIN. Steps where a fault is active but has a smaller effect (soiling that has
just started, shading when the sun is elsewhere, a derated inverter on a cloudy day) are left out of the point
metrics instead of being counted as normal.

Per-type precision is computed on healthy points plus points of that type, so all types share the same false
positives; the 'all' row pools every type. Because the threshold is set on calibration days, the false-positive rate
on healthy test steps differs between methods; `matched_recall` therefore also compares the methods at one and the
same false-positive rate on the healthy test steps.

An event counts as detected when at least DETECT_SHARE of its labelled steps raise an alarm. A single alarmed step
is not enough: methods that alarm on 5-18 % of all daylight steps would "detect" most long events by coincidence.
`placebo_events` measures that coincidence by shifting the alarms by whole days within the test period.

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
DETECT_SHARE = 0.5  # share of an event's labelled steps that must raise an alarm
PLACEBO_DAYS = (29, 85, 141, 196, 252, 308, 364, 420, 476, 531, 587, 643)  # whole-day shifts of the alarm series


def ground_truth(df: pd.DataFrame, daylight) -> pd.DataFrame:
    active = (df["fault"] != "").to_numpy()
    label = active & (df["impact"] >= IMPACT_MIN).to_numpy()
    daylight = np.asarray(daylight, bool)
    return pd.DataFrame({"daylight": daylight, "active": active, "label": label,
                         "kind": df["fault"].where(label, "").to_numpy(),
                         "evaluated": daylight & (~active | label), "impact": df["impact"].to_numpy()},
                        index=df.index)


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


def _flagged(alarm: np.ndarray, events: pd.DataFrame, index: pd.DatetimeIndex,
             labelled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Number of labelled steps and of alarmed labelled steps per event."""
    n, hit = np.zeros(len(events), int), np.zeros(len(events), int)
    for i, ev in enumerate(events.itertuples()):
        inside = (index >= ev.start) & (index < ev.end) & labelled
        n[i], hit[i] = inside.sum(), (inside & alarm).sum()
    return n, hit


def event_metrics(method: str, alarm: pd.Series, events: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    """Per injected event: share of its labelled daylight steps that raised an alarm, detected (at least
    DETECT_SHARE of them; events without a labelled step cannot be detected), and for detected events the time
    from the event start to the first alarmed labelled step and the fault's impact at that step."""
    labelled = truth["label"].to_numpy() & truth["daylight"].to_numpy()
    a = alarm.to_numpy(bool)
    n, hit = _flagged(a, events, alarm.index, labelled)
    rows = []
    for i, ev in enumerate(events.itertuples()):
        detected = bool(n[i]) and hit[i] >= DETECT_SHARE * n[i]
        ttd, impact = np.nan, np.nan
        if detected:
            first = np.flatnonzero((alarm.index >= ev.start) & (alarm.index < ev.end) & labelled & a)[0]
            ttd = (alarm.index[first] - ev.start) / pd.Timedelta("1h")
            impact = float(truth["impact"].iloc[first])
        rows.append({"method": method, "fault_id": ev.fault_id, "fault": ev.kind, "severity": ev.severity,
                     "labelled_steps": int(n[i]), "flagged_share": hit[i] / n[i] if n[i] else np.nan,
                     "any_hit": bool(hit[i]), "detected": detected, "ttd_hours": ttd,
                     "impact_at_detection": impact})
    return rows


def placebo_events(method: str, alarm: pd.Series, events: pd.DataFrame, truth: pd.DataFrame, mask,
                   shifts_days=PLACEBO_DAYS) -> dict:
    """Events 'detected' by chance: the alarm series of the test period (`mask`, contiguous) is shifted circularly
    by whole days, which keeps its daily and seasonal pattern but breaks its link to the faults."""
    m = np.flatnonzero(np.asarray(mask, bool))
    labelled = (truth["label"].to_numpy() & truth["daylight"].to_numpy())[m]
    a, index = alarm.to_numpy(bool)[m], alarm.index[m]
    steps_per_day = int(pd.Timedelta("1D") / (index[1] - index[0]))
    det, anyhit = [], []
    for d in shifts_days:
        n, hit = _flagged(np.roll(a, d * steps_per_day), events, index, labelled)
        det.append(int(((n > 0) & (hit >= DETECT_SHARE * n)).sum()))
        anyhit.append(int((hit > 0).sum()))
    return {"method": method, "chance_detected_mean": float(np.mean(det)), "chance_detected_max": int(np.max(det)),
            "chance_any_hit_mean": float(np.mean(anyhit)), "chance_any_hit_max": int(np.max(anyhit))}


def precision_by_type(score: pd.Series, truth: pd.DataFrame, mask) -> dict[str, float]:
    """Average precision, pooled and per type (healthy steps plus the steps of that type)."""
    m = np.asarray(mask, bool) & truth["evaluated"].to_numpy()
    s = score.to_numpy(float)
    label, kind = truth["label"].to_numpy(), truth["kind"].to_numpy()
    out = {}
    for k in ("all", *FAULT_TYPES):
        pos = m & label if k == "all" else m & (kind == k)
        sel = (m & ~label) | pos
        out[k] = average_precision(s[sel], pos[sel])
    return out


def matched_recall(method: str, score: pd.Series, truth: pd.DataFrame, mask, fpr: float = 0.005) -> list[dict]:
    """Recall, pooled and per type, at the threshold that flags `fpr` of the healthy evaluated test steps."""
    m = np.asarray(mask, bool) & truth["evaluated"].to_numpy()
    s = score.to_numpy(float)
    label, kind = truth["label"].to_numpy(), truth["kind"].to_numpy()
    thr = np.nanquantile(s[m & ~label], 1 - fpr)
    rows = []
    for k in ("all", *FAULT_TYPES):
        pos = m & label if k == "all" else m & (kind == k)
        rows.append({"method": method, "fault": k, "recall_matched": float((s[pos] > thr).mean()) if pos.any()
                     else np.nan})
    return rows


def false_alarms(method: str, alarm: pd.Series, truth: pd.DataFrame, mask, sky: pd.Series | None = None) -> dict:
    """Alarm episodes (runs of consecutive alarmed daylight steps) that start while no fault is active; with the
    simulated `sky` regime, also how many of them start on a broken-cloud day."""
    m = np.asarray(mask, bool) & truth["daylight"].to_numpy()
    seq = alarm.to_numpy(bool)[m]
    starts = seq & ~np.r_[False, seq[:-1]]
    healthy = ~truth["active"].to_numpy()[m]
    healthy_days = (np.asarray(mask, bool) & ~truth["active"].to_numpy()).sum() * 0.25 / 24
    n = int((starts & healthy).sum())
    out = {"method": method, "false_alarm_episodes": n, "healthy_days": round(float(healthy_days), 1),
           "per_30_days": n / healthy_days * 30 if healthy_days else np.nan}
    if sky is not None:
        out["on_broken_cloud_days"] = int((starts & healthy & (sky.to_numpy()[m] == "broken")).sum())
    return out


def summarize_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Per method and fault type: events, detectable events (with labelled steps), detected events, events with
    any alarmed labelled step, median time to detection and median impact at detection (detected events only)."""
    g = ev.groupby(["method", "fault"], sort=False)
    return pd.DataFrame({"events": g.size(), "detectable": g["labelled_steps"].apply(lambda x: int((x > 0).sum())),
                         "detected": g["detected"].sum(), "any_hit": g["any_hit"].sum(),
                         "median_ttd_hours": g["ttd_hours"].median(),
                         "median_impact_at_detection": g["impact_at_detection"].median()}).reset_index()
