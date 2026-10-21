"""Command line interface.

Usage:
    python -m pv_anomaly simulate --start 2021-01-01 --end 2024-01-01 --train-end 2022-01-01 --seed 42
    python -m pv_anomaly run                                  # simulate, train, evaluate, write results and charts
    python -m pv_anomaly run --no-train                       # score with the saved models/autoencoder.pt
    python -m pv_anomaly sweep --seeds 1 2 3 4 5              # repeat the experiment on other simulated plants
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

from pv_anomaly import plots
from pv_anomaly.autoencoder import AEConfig, AutoencoderDetector
from pv_anomaly.baselines import BASELINES
from pv_anomaly.detect import calibrate, smooth
from pv_anomaly.evaluate import (event_metrics, false_alarms, ground_truth, point_metrics, pr_curve,
                                 summarize_events)
from pv_anomaly.features import build_features, split_masks
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pv-anomaly", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
    {"simulate": cmd_simulate, "run": cmd_run, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
