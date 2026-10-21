import numpy as np
import pandas as pd
import pytest

from pv_anomaly.detect import calibrate, smooth
from pv_anomaly.evaluate import (average_precision, event_metrics, false_alarms, ground_truth, point_metrics,
                                 summarize_events)

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
    alarm.iloc[[1, 6, 8, 11]] = True  # 1: before event 0; 8: soiling not labelled yet; 11: after event 1 ended
    ev = pd.DataFrame(event_metrics("m", alarm, events, truth))
    assert ev["detected"].tolist() == [True, False]
    assert ev["ttd_hours"].iloc[0] == pytest.approx(0.5)
    s = summarize_events(ev)
    assert s["detection_rate"].tolist() == [1.0, 0.0]


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
