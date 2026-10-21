import numpy as np
import pandas as pd

from pv_anomaly.features import FEATURES, MIN_ELEVATION, NOMINAL_PR, build_features, make_windows, split_masks
from pv_anomaly.plant import PlantConfig, solar_geometry

PLANT = PlantConfig()


def test_make_windows_order_and_alignment():
    X = np.arange(20, dtype=float).reshape(10, 2)  # row i = [2i, 2i+1]
    W, end = make_windows(X, 3)
    assert W.shape == (8, 6)
    assert list(end) == list(range(2, 10))
    assert W[0].tolist() == [0, 1, 2, 3, 4, 5]  # time-major: rows 0, 1, 2
    assert W[-1].tolist() == X[7:10].ravel().tolist()


def test_make_windows_drops_nan():
    X = np.ones((10, 2))
    X[5, 1] = np.nan
    W, end = make_windows(X, 3)
    assert not np.isnan(W).any()
    assert set(end) == {2, 3, 4, 8, 9}


def test_build_features():
    index = pd.date_range("2021-06-21", periods=96, freq="15min", tz="UTC")
    geo = solar_geometry(index, PLANT)
    poa = geo["poa_clearsky"].to_numpy()
    df = pd.DataFrame({"p_ac": 0.75 * PLANT.pdc0_kw * poa / 1000, "poa": poa, "t_mod": 30.0, "t_amb": 20.0},
                      index=index)
    f = build_features(df, PLANT)
    assert list(f.columns) == [*FEATURES, "daylight"]
    sunny = poa >= 50
    assert np.allclose(f.loc[sunny, "pr"], 0.75)
    assert (f.loc[~sunny, "pr"] == NOMINAL_PR).all()
    assert np.allclose(f.loc[sunny, "kc"], 1.0)
    assert (f["daylight"] == (geo["elevation"] >= MIN_ELEVATION)).all()


def test_split_masks_whole_days():
    index = pd.date_range("2021-01-01", "2022-12-31", freq="15min", tz="Europe/Warsaw").tz_convert("UTC")
    m = split_masks(index, "Europe/Warsaw", "2022-01-01", val_frac=0.2, seed=0)
    assert ((m["train"].astype(int) + m["val"] + m["test"]) == 1).all()
    local = index.tz_convert("Europe/Warsaw")
    assert (local[m["test"]] >= pd.Timestamp("2022-01-01", tz="Europe/Warsaw")).all()
    val_days = pd.Series(m["val"], index=local.normalize()).groupby(level=0).agg(["min", "max"])
    assert (val_days["min"] == val_days["max"]).all()  # a day is entirely in or out of validation
    assert abs(val_days["max"].iloc[:365].mean() - 0.2) < 0.01
