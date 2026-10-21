"""Static charts (PNG) for the README.

Usage:
    from pv_anomaly import plots
    plots.f1_heatmap(points, Path("docs/img/f1_by_fault.png"))
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pv_anomaly.simulate import FAULT_TYPES  # noqa: E402

TZ = "Europe/Warsaw"
# one fixed colour per method, used in every chart
METHODS = {"autoencoder": ("Autoencoder (PR residual)", "#2a78d6"),
           "regression": ("Regression (ASTM E2848)", "#eb6834"),
           "physical": ("Physical model (pvlib)", "#1baf7a"),
           "pr_threshold": ("PR threshold", "#eda100"),
           "autoencoder_mse": ("Autoencoder (MSE)", "#e87ba4")}
INK, MUTED, GRID, ALARM = "#0b0b0b", "#52514e", "#e4e3df", "#f6c3c3"
FAULT_LABELS = {k: k.replace("_", " ") for k in FAULT_TYPES}
FAULT_CODES = {"soiling": "SL", "shading": "SH", "string_outage": "ST", "inverter_derate": "INV", "sensor_drift": "SD"}

plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
                     "legend.frameon": False})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _label(method: str) -> str:
    return METHODS.get(method, (method, None))[0]


def fault_examples(df: pd.DataFrame, events: pd.DataFrame, alarm: pd.Series, path: Path, days: float = 3.0):
    """One panel per fault type: measured vs. healthy AC power around the most severe part of an event,
    with the autoencoder's alarms shaded."""
    fig, axes = plt.subplots(len(FAULT_TYPES), 1, figsize=(10, 2.1 * len(FAULT_TYPES)))
    for ax, kind in zip(axes, FAULT_TYPES):
        evs = events[events["kind"] == kind]
        if evs.empty:
            ax.set_visible(False)
            continue
        ev = evs.loc[evs["severity"].abs().idxmax()]
        # last `days` of the event (gradual faults are strongest at the end), widened to whole local days
        local_end = ev["end"].tz_convert(TZ).normalize() + pd.Timedelta("1D")
        t0, t1 = (local_end - pd.Timedelta(days=days)).tz_convert("UTC"), local_end.tz_convert("UTC")
        w = df.loc[t0:t1]
        x = w.index.tz_convert(TZ).tz_localize(None)
        a = alarm.loc[t0:t1].to_numpy(bool)
        ax.fill_between(x, 0, 1, where=a, transform=ax.get_xaxis_transform(), color=ALARM, lw=0, step="mid",
                        label="autoencoder alarm")
        ax.plot(x, w["p_ac_healthy"], color="#9a9893", lw=1.2, label="healthy (ground truth)")
        ax.plot(x, w["p_ac"], color=METHODS["autoencoder"][1], lw=1.2, label="measured")
        start_local = ev["start"].tz_convert(TZ).strftime("%d %b %Y")
        note = " - the plant is healthy, the irradiance sensor is not" if kind == "sensor_drift" else ""
        ax.set_title(f"{FAULT_LABELS[kind]}: event from {start_local}, severity {ev['severity']:+.2f}{note}",
                     loc="left", fontsize=9)
        ax.set_ylabel("AC power [kW]")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Last days of one event per fault type (local time)", x=0.01, ha="left", fontsize=10)
    _save(fig, path)


def f1_heatmap(points: pd.DataFrame, path: Path, value: str = "f1"):
    """Point-wise F1 per method (rows) and fault type (columns), numbers printed in the cells."""
    order = [m for m in METHODS if m in set(points["method"])]
    cols = ["all", *FAULT_TYPES]
    t = points.pivot(index="method", columns="fault", values=value).loc[order, cols]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.2))
    ax.pcolormesh(np.arange(len(cols) + 1) - 0.5, np.arange(len(order) + 1) - 0.5, t.to_numpy(float),
                  cmap="Blues", vmin=0, vmax=1, edgecolors="white", linewidth=2)
    ax.invert_yaxis()
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(cols)), ["all faults", *[FAULT_LABELS[c] for c in FAULT_TYPES]])
    ax.set_yticks(range(len(order)), [_label(m) for m in order])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Point-wise {value.upper() if value == 'f1' else value} on the test period (higher is better)",
                 loc="left")
    _save(fig, path)


def time_to_detection(ev: pd.DataFrame, path: Path, methods=("autoencoder", "regression", "physical")):
    """Time from fault start to the first alarm, one dot per event; missed events drawn at the right edge."""
    ev = ev[ev["method"].isin(methods)]
    finite = ev["ttd_hours"].dropna() / 24
    xmax = max(30.0, float(finite.max()) * 1.15) if len(finite) else 30.0
    fig, ax = plt.subplots(figsize=(9, 3.6))
    n = len(methods)
    for j, m in enumerate(methods):
        e = ev[ev["method"] == m]
        y = np.array([FAULT_TYPES.index(k) for k in e["fault"]]) + (j - (n - 1) / 2) * 0.22
        days = (e["ttd_hours"] / 24).to_numpy()
        hit = np.isfinite(days)
        ax.scatter(days[hit], y[hit], s=36, color=METHODS[m][1], edgecolor="white", linewidth=1, zorder=3,
                   label=_label(m))
        ax.scatter(np.full((~hit).sum(), xmax), y[~hit], s=40, marker="x", color=METHODS[m][1], zorder=3)
    ax.axvline(xmax, color=GRID, lw=1)
    ax.text(xmax, len(FAULT_TYPES) - 0.45, "missed ", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(FAULT_TYPES)), [FAULT_LABELS[k] for k in FAULT_TYPES])
    ax.set_ylim(len(FAULT_TYPES) - 0.5, -0.5)
    ax.set_xlim(-0.5, xmax * 1.02)
    ax.set_xlabel("Days from fault start to first alarm")
    ax.set_title("Time to detection per injected event", loc="left", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(methods), fontsize=8)
    _save(fig, path)


def score_timeline(scores: dict[str, pd.Series], thresholds: dict[str, float], events: pd.DataFrame,
                   test_mask, path: Path, methods=("autoencoder", "regression")):
    """Daily maximum of the smoothed score divided by its threshold (alarm above 1), faults shaded."""
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 2.3 * len(methods)), sharex=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        s = (scores[m][np.asarray(test_mask, bool)] / thresholds[m]).tz_convert(TZ)
        daily = s.groupby(s.index.tz_localize(None).normalize()).max()
        for ev in events.itertuples():
            ax.axvspan(ev.start.tz_convert(TZ).tz_localize(None), ev.end.tz_convert(TZ).tz_localize(None),
                       color="#ecebe7", lw=0)
        ax.plot(daily.index, daily.to_numpy(), color=METHODS[m][1], lw=1.0)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 4)  # headroom for the fault codes
        ax.set_ylabel("score / threshold")
        ax.set_title(_label(m), loc="left")
        for ev in events.itertuples():
            mid = ev.start.tz_convert(TZ).tz_localize(None) + (ev.end - ev.start) / 2
            ax.text(mid, 0.97, FAULT_CODES[ev.kind], transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.5, color=MUTED)
    fig.text(0.01, 0.005, "Grey spans: injected faults (SL soiling, SH shading, ST string outage, INV inverter derate, "
             "SD sensor drift). Daily maximum of the smoothed score; alarms above the black line.",
             fontsize=7.5, color=MUTED)
    _save(fig, path)


def pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], ap: dict[str, float], path: Path):
    """Pooled precision-recall curves (threshold-free comparison)."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m in METHODS:
        if m in curves:
            r, p = curves[m]
            ax.plot(r, p, color=METHODS[m][1], lw=1.6, label=f"{_label(m)}  AP {ap[m]:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall, all fault types pooled", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def real_timeline(daily: pd.DataFrame, path: Path, title: str, methods=("autoencoder", "regression")):
    """A real system's test year: daily performance ratio, then each method's daily peak score / threshold.
    Light grey days: AC power missing; darker grey: no production in bright sun."""
    fig, axes = plt.subplots(1 + len(methods), 1, figsize=(10, 1.9 * (1 + len(methods)) + 0.5), sharex=True)
    x = daily.index.tz_localize(None)
    one = pd.Timedelta("1D")
    for ax in axes:
        for d in x[(daily["missing_share"] > 0.5).to_numpy()]:
            ax.axvspan(d, d + one, color="#ecebe7", lw=0)
        for d in x[daily["dead"].to_numpy()]:
            ax.axvspan(d, d + one, color="#cfcdc6", lw=0)
    axes[0].plot(x, daily["pr"], ".", ms=3, color=MUTED)
    axes[0].set_ylabel("daily PR")
    axes[0].set_ylim(-0.03, max(0.8, float(daily["pr"].max()) + 0.05))
    axes[0].set_title(title, loc="left")
    status = x[(daily["status_max"] > 0).to_numpy() & ~(daily["missing_share"] > 0.5).to_numpy()]
    axes[0].plot(status, np.full(len(status), 0.02), "|", ms=8, color=INK, label="inverter error code reported")
    axes[0].legend(loc="lower left", fontsize=8)
    for ax, m in zip(axes[1:], methods):
        ax.plot(x, daily[f"{m}_peak"], color=METHODS[m][1], lw=1.0)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("peak score / threshold")
        ax.set_title(_label(m), loc="left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.text(0.01, 0.005, "Darker grey: days without production in bright sun; light grey: no AC power data. "
             "Alarms above the black line.", fontsize=7.5, color=MUTED)
    _save(fig, path)


def real_examples(df: pd.DataFrame, expected: pd.Series, alarm: pd.Series, windows: list[tuple[pd.Timestamp, str]],
                  tz: str, path: Path, days: int = 2):
    """Measured vs. expected AC power on selected days of a real system, autoencoder alarms shaded."""
    fig, axes = plt.subplots(len(windows), 1, figsize=(10, 2.2 * len(windows)), squeeze=False)
    for ax, (start, title) in zip(axes[:, 0], windows):
        t0 = start.tz_convert(tz).normalize()
        sel = (df.index >= t0) & (df.index < t0 + pd.Timedelta(days=days))
        x = df.index[sel].tz_convert(tz).tz_localize(None)
        ax.fill_between(x, 0, 1, where=alarm.to_numpy(bool)[sel], transform=ax.get_xaxis_transform(), color=ALARM,
                        lw=0, step="mid", label="autoencoder alarm")
        ax.plot(x, expected[sel], color="#9a9893", lw=1.2, label="expected (regression on POA, module temp.)")
        ax.plot(x, df["p_ac"][sel], color=METHODS["autoencoder"][1], lw=1.2, label="measured")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_ylabel("AC power [kW]")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Local standard time", x=0.01, ha="left", fontsize=9, color=MUTED)
    _save(fig, path)
