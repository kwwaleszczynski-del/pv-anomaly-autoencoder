"""Command line interface.

Usage:
    python -m pv_anomaly simulate --start 2021-01-01 --end 2024-01-01 --train-end 2022-01-01 --seed 42
    python -m pv_anomaly run                                  # simulate, train, evaluate, write results and charts
    python -m pv_anomaly run --no-train                       # score with the saved models/autoencoder.pt
    python -m pv_anomaly sweep --seeds 1 2 3 4 5              # repeat the experiment on other simulated plants
    python -m pv_anomaly real --system 34                     # real data: NREL PVDAQ system 34 (downloads ~8 MB)
    python -m pv_anomaly run --help

The first year is healthy and used for training (80 % of days) and threshold calibration (20 % of days, never
trained on); faults are injected from --train-end on and everything after it is the test period.
Data go to ./data (override with --data-dir or PV_ANOMALY_DATA_DIR), results to ./results, charts to ./docs/img
and the trained model to ./models.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pv_anomaly import plots, pvdaq
from pv_anomaly.autoencoder import AEConfig, AutoencoderDetector
from pv_anomaly.baselines import BASELINES, PhysicalBaseline, RegressionBaseline
from pv_anomaly.detect import calibrate, smooth
from pv_anomaly.evaluate import (event_metrics, false_alarms, ground_truth, point_metrics, pr_curve,
                                 summarize_events)
from pv_anomaly.features import FEATURES, build_features, split_masks
from pv_anomaly.plant import PlantConfig
from pv_anomaly.simulate import simulate

log = logging.getLogger("pv_anomaly")


def cmd_simulate(args, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, events = simulate(PlantConfig(), start=args.start, end=args.end, fault_start=args.train_end,
                          per_type=args.per_type, seed=args.seed)
    log.info("simulated %d steps %s .. %s, %d fault events", len(df), df.index[0], df.index[-1], len(events))
    if save:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.data_dir / "synthetic.parquet")
        events.to_csv(args.data_dir / "fault_events.csv", index=False)
    return df, events


def experiment(args, df: pd.DataFrame, events: pd.DataFrame, ae: AutoencoderDetector | None = None) -> dict:
    """Fit all detectors on the healthy year, calibrate thresholds on validation days, evaluate on the test period."""
    plant = PlantConfig()
    feats = build_features(df, plant)
    masks = split_masks(df.index, plant.tz, args.train_end, val_frac=0.2, seed=args.seed)
    day = feats["daylight"].to_numpy()
    truth = ground_truth(df, day)

    raw, signed = {}, {}
    for B in BASELINES:
        model = B(plant).fit(df, masks["train"] & day)
        raw[model.name], signed[model.name] = model.residual_z(df), True

    history, t_train = [], time.time()
    if ae is None:
        ae = AutoencoderDetector(AEConfig(window=args.window, latent=args.latent, epochs=args.epochs, seed=args.seed))
        history = ae.fit(feats, masks["train"], masks["val"])
    t_train = time.time() - t_train
    raw["autoencoder"], signed["autoencoder"] = ae.score(feats, "pr_residual"), True
    raw["autoencoder_mse"], signed["autoencoder_mse"] = ae.score(feats, "mse"), False

    val_day = masks["val"] & day
    test_eval = masks["test"] & truth["evaluated"].to_numpy()
    res = {"ae": ae, "history": history, "train_seconds": t_train, "masks": masks, "truth": truth, "scores": {},
           "thresholds": {}, "alarms": {}, "curves": {}}
    pts, evs, fas = [], [], []
    for name in raw:
        s = smooth(raw[name], day, args.smooth, signed[name])
        thr = calibrate(s, val_day, args.quantile)
        alarm = (s > thr).fillna(False)
        res["scores"][name], res["thresholds"][name], res["alarms"][name] = s, thr, alarm
        pts += point_metrics(name, alarm, s, truth, masks["test"])
        evs += event_metrics(name, alarm, events, truth)
        fas.append(false_alarms(name, alarm, truth, masks["test"]))
        res["curves"][name] = pr_curve(s.to_numpy()[test_eval], truth["label"].to_numpy()[test_eval])
    res["points"], res["events"], res["false_alarms"] = pd.DataFrame(pts), pd.DataFrame(evs), pd.DataFrame(fas)
    res["summary"] = summarize_events(res["events"])
    return res


def headline(res: dict) -> pd.DataFrame:
    """One row per method: pooled point metrics, events detected, false alarms."""
    p = res["points"][res["points"]["fault"] == "all"].set_index("method")
    s = res["summary"].groupby("method")[["detected", "events"]].sum()
    fa = res["false_alarms"].set_index("method")["per_30_days"]
    return pd.DataFrame({"precision": p["precision"], "recall": p["recall"], "f1": p["f1"],
                         "avg_precision": p["avg_precision"], "events_detected": s["detected"],
                         "events": s["events"], "false_alarms_per_30d": fa}).loc[p.index]


def cmd_run(args):
    t_start = time.time()
    torch.manual_seed(args.seed)
    df, events = cmd_simulate(args)
    ae = None
    if args.no_train:
        ae = AutoencoderDetector.load(args.model)
        log.info("loaded autoencoder from %s", args.model)
    res = experiment(args, df, events, ae)
    if not args.no_train:
        res["ae"].save(args.model, threshold=res["thresholds"]["autoencoder"], smooth=args.smooth,
                       quantile=args.quantile, seed=args.seed)

    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "fault_events.csv", index=False)
    res["points"].round(4).to_csv(out / "metrics_points.csv", index=False)
    res["summary"].round(2).to_csv(out / "metrics_events.csv", index=False)
    res["events"].round(2).to_csv(out / "events_detail.csv", index=False)
    res["false_alarms"].round(3).to_csv(out / "false_alarms.csv", index=False)
    if res["history"]:
        pd.DataFrame(res["history"]).round(5).to_csv(out / "training_history.csv", index=False)
    head = headline(res)
    head.round(3).to_csv(out / "headline.csv")
    for name, r in head.iterrows():
        log.info("%-16s F1 %.2f  P %.2f  R %.2f  AP %.2f  events %d/%d  false alarms %.2f per 30 days", name,
                 r["f1"], r["precision"], r["recall"], r["avg_precision"], r["events_detected"], r["events"],
                 r["false_alarms_per_30d"])

    img = args.img_dir
    ap = head["avg_precision"].to_dict()
    plots.fault_examples(df, events, res["alarms"]["autoencoder"], img / "fault_examples.png")
    plots.f1_heatmap(res["points"], img / "f1_by_fault.png")
    plots.time_to_detection(res["events"], img / "time_to_detection.png")
    plots.score_timeline(res["scores"], res["thresholds"], events, res["masks"]["test"], img / "score_timeline.png")
    plots.pr_curves(res["curves"], ap, img / "pr_curves.png")

    cfg, masks, truth = res["ae"].config, res["masks"], res["truth"]
    day = truth["daylight"].to_numpy()
    (out / "run.json").write_text(json.dumps({
        "seed": args.seed, "start": args.start, "end": args.end, "train_end": args.train_end,
        "autoencoder": {**cfg.__dict__, "hidden": list(cfg.hidden), "epochs_run": len(res["history"]),
                        "train_seconds": round(res["train_seconds"], 1), "trained": not args.no_train},
        "smooth": args.smooth, "quantile": args.quantile, "thresholds": res["thresholds"],
        "steps": {"train": int((masks["train"] & day).sum()), "val": int((masks["val"] & day).sum()),
                  "test_evaluated": int((masks["test"] & truth["evaluated"].to_numpy()).sum()),
                  "test_positive": int((masks["test"] & truth["label"].to_numpy()).sum())},
        "events": int(len(events)), "runtime_seconds": round(time.time() - t_start, 1),
        "torch": torch.__version__, "numpy": np.__version__,
    }, indent=2, default=str))
    log.info("results in %s, charts in %s, model in %s (%.0f s)", out, img, args.model, time.time() - t_start)


def cmd_sweep(args):
    """Repeat the whole experiment for several seeds (new weather, faults, split and network initialisation)."""
    rows = []
    for seed in args.seeds:
        args.seed = seed
        torch.manual_seed(seed)
        df, events = cmd_simulate(args, save=False)
        head = headline(experiment(args, df, events)).reset_index()
        head.insert(0, "seed", seed)
        rows.append(head)
        log.info("seed %d: autoencoder F1 %.2f, regression F1 %.2f", seed,
                 head.set_index("method").loc["autoencoder", "f1"], head.set_index("method").loc["regression", "f1"])
    allrows = pd.concat(rows, ignore_index=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    allrows.round(4).to_csv(args.results_dir / "sweep_seeds.csv", index=False)
    stats = allrows.drop(columns="seed").groupby("method", sort=False).agg(["mean", "std"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    stats.round(3).to_csv(args.results_dir / "sweep_summary.csv")
    log.info("sweep over seeds %s:\n%s", args.seeds,
             stats[["f1_mean", "f1_std", "avg_precision_mean", "avg_precision_std"]].round(2).to_string())


def _episodes(alarm: np.ndarray, ratio: pd.Series, mask: np.ndarray, outage: np.ndarray) -> pd.DataFrame:
    """Alarm episodes = runs of consecutive alarmed steps among `mask` steps, with peak score / threshold."""
    idx = np.flatnonzero(mask)
    seq = alarm[idx]
    run = np.cumsum(seq & ~np.r_[False, seq[:-1]])
    rows = []
    for k in np.unique(run[seq]):
        pos = idx[seq & (run == k)]
        rows.append({"start": ratio.index[pos[0]], "end": ratio.index[pos[-1]], "steps": len(pos),
                     "peak_ratio": float(ratio.iloc[pos].max()), "outage": bool(outage[pos[0]:pos[-1] + 1].any())})
    return pd.DataFrame(rows, columns=["start", "end", "steps", "peak_ratio", "outage"])


def cmd_real(args):
    """Learn normal behaviour from one year of a real PVDAQ system and score another year. There are no fault
    labels, so the output is the list of alarm episodes, and how fast each detector flags the days on which the
    inverter produced nothing in full sun compared with the inverter's own error code."""
    t_start = time.time()
    torch.manual_seed(args.seed)
    plant = pvdaq.SYSTEMS[args.system]
    files = pvdaq.fetch(args.system, args.start, args.end, args.data_dir / "pvdaq")
    df = pvdaq.load(files, plant)
    feats = build_features(df, plant)
    day = feats["daylight"].to_numpy()
    masks = split_masks(df.index, plant.tz, args.train_end, val_frac=0.2, seed=args.seed,
                        train_start=args.train_start)
    test = masks["test"] & day

    # this site's ambient temperature sensor and anemometer are unreliable (long runs of 0 degF, stuck at 0 mph):
    # the regression uses module temperature and no wind, the autoencoder drops the ambient temperature feature
    reg = RegressionBaseline(plant, temp="t_mod", wind=False)
    raw = {}
    for model in (reg, PhysicalBaseline(plant)):
        raw[model.name] = model.fit(df, masks["train"] & day).residual_z(df)
    ae = AutoencoderDetector(AEConfig(window=args.window, latent=args.latent, epochs=args.epochs, seed=args.seed),
                             features=tuple(f for f in FEATURES if f != "t_amb"))
    t_train = time.time()
    history = ae.fit(feats, masks["train"], masks["val"])
    t_train = time.time() - t_train
    raw["autoencoder"] = ae.score(feats)

    local_day = df.index.tz_convert(plant.tz).normalize()
    bright = df["poa"] > 50
    g = pd.DataFrame({"irr": (df["poa"].where(bright & df["p_ac"].notna()) * 0.25 / 1000),
                      "energy": df["p_ac"].where(bright) * 0.25, "missing": df["p_ac"].isna(),
                      "status": df["status"]}).groupby(local_day)
    daily = pd.DataFrame({"irradiation_kwh_m2": g["irr"].sum(), "energy_kwh": g["energy"].sum(min_count=1),
                          "missing_share": g["missing"].mean(), "status_max": g["status"].max()})
    daily["pr"] = (daily["energy_kwh"] / (plant.pdc0_kw * daily["irradiation_kwh_m2"])).where(
        daily["irradiation_kwh_m2"] > 1)
    # days without production although the sun was shining (the outage this site had in June 2016); the outage
    # starts at the first bright step of the first such day
    daily["dead"] = daily["pr"] < 0.05
    dead = (test & daily["dead"].reindex(local_day).fillna(False).to_numpy()
            & (df["poa"] > 300).to_numpy() & (df["p_ac"] < 0.01 * plant.pac0_kw).to_numpy())
    first_dead = df.index[dead][0] if dead.any() else None
    outage = np.zeros(len(df), bool)
    first_status = None
    if first_dead is not None:
        outage[(df.index >= first_dead) & (df.index <= df.index[dead][-1])] = True
        after = df.index[(df["status"] > 0).to_numpy() & (df.index >= first_dead)]
        first_status = after[0] if len(after) else None

    summary, episodes, ratio, alarms, thresholds = [], [], {}, {}, {}
    for name, r in raw.items():
        s = smooth(r, day, args.smooth, signed=True)
        thr = calibrate(s, masks["val"] & day, args.quantile)
        a = (s > thr).fillna(False).to_numpy() & test
        alarms[name], ratio[name], thresholds[name] = a, s / thr, thr
        ep = _episodes(a, ratio[name], test, outage).assign(method=name)
        episodes.append(ep)
        flagged = df.index[a & (df.index >= first_dead)] if first_dead is not None else []
        summary.append({"method": name, "threshold": thr, "alarm_episodes": len(ep),
                        "episodes_outside_outage": int((~ep["outage"]).sum()), "alarm_hours": a.sum() / 4,
                        "hours_to_flag_outage": (flagged[0] - first_dead) / pd.Timedelta("1h") if len(flagged)
                        else np.nan})
    summary.append({"method": "inverter_status_code", "hours_to_flag_outage":
                    (first_status - first_dead) / pd.Timedelta("1h") if first_status is not None else np.nan})
    summary = pd.DataFrame(summary)
    episodes = pd.concat(episodes, ignore_index=True)

    for name in raw:
        daily[f"{name}_peak"] = ratio[name].where(test).groupby(local_day).max()
    daily = daily.loc[np.unique(local_day[masks["test"]])]

    out = args.results_dir / f"pvdaq_{args.system}"
    out.mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(out / "summary.csv", index=False)
    episodes.round(3).to_csv(out / "episodes.csv", index=False)
    daily.round(4).to_csv(out / "daily.csv")
    ae.save(args.model, threshold=thresholds["autoencoder"], smooth=args.smooth, quantile=args.quantile,
            seed=args.seed, system=args.system)
    for r in summary.itertuples():
        log.info("%-20s episodes %s (%s outside the outage), hours to flag the outage %.2f", r.method,
                 r.alarm_episodes, r.episodes_outside_outage, r.hours_to_flag_outage)

    ae_ep = episodes[(episodes["method"] == "autoencoder") & ~episodes["outage"]].nlargest(2, "peak_ratio")
    windows = []
    if first_dead is not None and first_status is not None:
        lag = (first_status - first_dead) / pd.Timedelta("1D")
        windows.append((first_dead - pd.Timedelta("1D"),
                        f"Outage onset: no output from the first morning, inverter error code {lag:.0f} days later"))
    for e in ae_ep.itertuples():
        windows.append((e.start - pd.Timedelta("1D"),
                        f"Autoencoder episode {e.start.tz_convert(plant.tz):%d %b %Y} "
                        f"(peak {e.peak_ratio:.1f} x threshold), previous day for reference"))
    years = "-".join(str(y) for y in sorted(set(local_day[masks["test"]].year)))
    plots.real_timeline(daily, args.img_dir / f"pvdaq{args.system}_timeline.png",
                        f"PVDAQ system {args.system}, scored period {years}: daily performance ratio")
    plots.real_examples(df, reg.expected(df), pd.Series(alarms["autoencoder"], index=df.index), windows,
                        plant.tz, args.img_dir / f"pvdaq{args.system}_examples.png")
    (out / "run.json").write_text(json.dumps({
        "system": args.system, "start": args.start, "end": args.end, "train": [args.train_start, args.train_end],
        "files": len(files), "megabytes": round(sum(f.stat().st_size for f in files) / 1e6, 2),
        "missing_ac_share": round(float(df["p_ac"].isna().mean()), 4),
        "first_no_output_step": str(first_dead), "first_status_code_after": str(first_status),
        "autoencoder": {"features": list(ae.features), "epochs_run": len(history), "train_seconds": round(t_train, 1),
                        "latent": args.latent, "window": args.window, "seed": args.seed},
        "smooth": args.smooth, "quantile": args.quantile, "thresholds": thresholds,
        "runtime_seconds": round(time.time() - t_start, 1),
    }, indent=2, default=str))
    log.info("results in %s, charts in %s (%.0f s)", out, args.img_dir, time.time() - t_start)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pv-anomaly", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("real", help="real data: NREL PVDAQ (CC BY 4.0)")
    p.add_argument("--system", type=int, choices=sorted(pvdaq.SYSTEMS), default=34)
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2017-12-31", help="last day (inclusive)")
    p.add_argument("--train-start", default="2017-01-01", help="healthy period used for training and validation")
    p.add_argument("--train-end", default="2018-01-01", help="day after the training period; other days are scored")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--latent", type=int, default=6)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--smooth", type=int, default=8)
    p.add_argument("--quantile", type=float, default=0.995)
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PV_ANOMALY_DATA_DIR", "data")))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--img-dir", type=Path, default=Path("docs/img"))
    p.add_argument("--model", type=Path, default=Path("models/autoencoder_pvdaq34.pt"))
    for name in ("simulate", "run", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--start", default="2021-01-01", help="first simulated day (local time)")
        p.add_argument("--end", default="2024-01-01", help="day after the last simulated day")
        p.add_argument("--train-end", default="2022-01-01", help="healthy training data before, faults from here")
        p.add_argument("--per-type", type=int, default=4, help="fault events per type")
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PV_ANOMALY_DATA_DIR", "data")))
        if name == "sweep":
            p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
        else:
            p.add_argument("--seed", type=int, default=42, help="seed for weather, faults, split and network")
        if name in ("run", "sweep"):
            p.add_argument("--window", type=int, default=8, help="autoencoder window, 15-min steps")
            p.add_argument("--latent", type=int, default=6, help="autoencoder bottleneck size")
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--smooth", type=int, default=8, help="daylight steps averaged before thresholding")
            p.add_argument("--quantile", type=float, default=0.995, help="threshold quantile on validation days")
            p.add_argument("--results-dir", type=Path, default=Path("results"))
        if name == "run":
            p.add_argument("--model", type=Path, default=Path("models/autoencoder.pt"))
            p.add_argument("--no-train", action="store_true", help="load --model instead of training")
            p.add_argument("--img-dir", type=Path, default=Path("docs/img"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    {"simulate": cmd_simulate, "run": cmd_run, "sweep": cmd_sweep, "real": cmd_real}[args.cmd](args)


if __name__ == "__main__":
    main()
