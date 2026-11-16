import numpy as np
import pandas as pd
import pytest

from pv_anomaly.detect import calibrate, smooth
from pv_anomaly.evaluate import (average_precision, event_metrics, false_alarms, ground_truth, matched_recall,
                                 placebo_events, point_metrics, summarize_events)

IDX = pd.date_range("2022-06-01", periods=12, freq="15min", tz="UTC")


@pytest.fixture
def truth():
    df = pd.DataFrame({"fault": [""] * 4 + ["string_outage"] * 4 + ["soiling"] * 4,
                       "impact": [0.0] * 4 + [0.1] * 4 + [0.01, 0.01, 0.05, 0.05]}, index=IDX)
    return ground_truth(df, np.ones(12, bool))


def test_ground_truth(truth):
    assert truth["label"].tolist() == [False] * 4 + [True] * 4 + [False, False, True, True]
    assert truth["evaluated"].tolist() == [True] * 8 + [False, False, True, True]  # weak soiling is ignored


def test_point_metrics(truth):
    alarm = pd.Series([False, True, False, False, True, True, True, False, True, True, False, True], index=IDX)
    rows = {r["fault"]: r for r in point_metrics("m", alarm, alarm.astype(float), truth, np.ones(12, bool))}
    # all: tp = 3 (outage) + 1 (soiling), fp = 1 (index 1; index 8 and 9 are ignored), fn = 1 + 1
    assert rows["all"]["precision"] == pytest.approx(4 / 5)
    assert rows["all"]["recall"] == pytest.approx(4 / 6)
    assert rows["string_outage"]["precision"] == pytest.approx(3 / 4)
    assert rows["string_outage"]["recall"] == pytest.approx(3 / 4)
    assert rows["soiling"]["recall"] == pytest.approx(1 / 2)
    assert rows["shading"]["positives"] == 0


def test_average_precision():
    y = np.array([1, 1, 0, 0], bool)
    assert average_precision(np.array([4, 3, 2, 1.0]), y) == 1.0
    assert average_precision(np.array([1, 2, 3, 4.0]), y) == pytest.approx((1 / 3 + 2 / 4) / 2)
    assert average_precision(np.array([4, np.nan, 2, 1.0]), y) == 1.0


def test_event_metrics_time_to_detection(truth):
    events = pd.DataFrame({"fault_id": [0, 1], "kind": ["string_outage", "soiling"], "severity": [0.1, 0.1],
                           "start": [IDX[4], IDX[8]], "end": [IDX[8], IDX[11]]})
    alarm = pd.Series(False, index=IDX)
    alarm.iloc[[1, 5, 6, 8, 11]] = True  # 1: before event 0; 8: soiling not labelled yet; 11: after event 1 ended
    ev = pd.DataFrame(event_metrics("m", alarm, events, truth))
    assert ev["labelled_steps"].tolist() == [4, 1]
    assert ev["flagged_share"].tolist() == [0.5, 0.0]
    assert ev["detected"].tolist() == [True, False] and ev["any_hit"].tolist() == [True, False]
    assert ev["ttd_hours"].iloc[0] == pytest.approx(0.25) and ev["impact_at_detection"].iloc[0] == pytest.approx(0.1)
    alarm.iloc[5] = False  # one of four labelled steps is not enough
    ev = pd.DataFrame(event_metrics("m", alarm, events, truth))
    assert ev["any_hit"].tolist() == [True, False] and ev["detected"].tolist() == [False, False]
    assert np.isnan(ev["ttd_hours"]).all()
    s = summarize_events(ev)
    assert s["detectable"].tolist() == [1, 1] and s["detected"].tolist() == [0, 0] and s["any_hit"].tolist() == [1, 0]


def test_placebo_events():
    idx = pd.date_range("2022-01-01", periods=20 * 4, freq="6h", tz="UTC")  # 20 days, 4 steps a day
    fault = np.where((idx >= idx[8]) & (idx < idx[12]), "shading", "")
    tr = ground_truth(pd.DataFrame({"fault": fault, "impact": (fault != "") * 0.2}, index=idx), np.ones(len(idx), bool))
    events = pd.DataFrame({"fault_id": [0], "kind": ["shading"], "severity": [0.5],
                           "start": [idx[8]], "end": [idx[12]]})
    exact = pd.Series(fault != "", index=idx)
    p = placebo_events("m", exact, events, tr, np.ones(len(idx), bool), shifts_days=(3, 7))
    assert p["chance_detected_mean"] == 0 and p["chance_any_hit_max"] == 0
    always = pd.Series(True, index=idx)
    p = placebo_events("m", always, events, tr, np.ones(len(idx), bool), shifts_days=(3, 7))
    assert p["chance_detected_mean"] == 1 and p["chance_detected_max"] == 1


def test_matched_recall():
    n = 1000
    idx = pd.date_range("2022-01-01", periods=n + 20, freq="15min", tz="UTC")
    fault = np.r_[np.full(n, ""), np.full(10, "soiling"), np.full(10, "shading")]
    tr = ground_truth(pd.DataFrame({"fault": fault, "impact": (fault != "") * 0.2}, index=idx), np.ones(len(idx), bool))
    score = pd.Series(np.r_[np.linspace(0, 1, n), np.full(10, 2.0), np.full(10, 0.5)], index=idx)
    rows = {r["fault"]: r["recall_matched"] for r in matched_recall("m", score, tr, np.ones(len(idx), bool))}
    assert rows["soiling"] == 1.0 and rows["shading"] == 0.0 and rows["all"] == 0.5
    assert np.isnan(rows["string_outage"])


def test_false_alarm_episodes(truth):
    alarm = pd.Series([True, True, False, True] + [True] * 4 + [False] * 4, index=IDX)
    fa = false_alarms("m", alarm, truth, np.ones(12, bool))
    assert fa["false_alarm_episodes"] == 2  # [0, 1] and [3, ...] start while healthy


def test_smooth_is_causal_and_daylight_only():
    raw = pd.Series([1.0, -1.0, 3.0, np.nan, 5.0, -100.0], index=IDX[:6])
    day = np.array([True, True, True, True, True, False])
    s = smooth(raw, day, k=2, signed=True)
    assert s.iloc[:5].tolist() == [1.0, 0.0, 1.0, 3.0, 5.0]
    assert np.isnan(s.iloc[5])
    assert calibrate(pd.Series(np.arange(101.0)), np.ones(101, bool), 0.9) == pytest.approx(90.0)


def test_false_alarms_by_sky(truth):
    alarm = pd.Series([True, False, True, False] + [False] * 8, index=IDX)
    sky = pd.Series(["broken"] * 2 + ["clear"] * 10, index=IDX)
    fa = false_alarms("m", alarm, truth, np.ones(12, bool), sky=sky)
    assert fa["false_alarm_episodes"] == 2 and fa["on_broken_cloud_days"] == 1
