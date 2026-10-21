import numpy as np
import pandas as pd
import pytest

from pv_anomaly.autoencoder import AEConfig, AutoencoderDetector
from pv_anomaly.features import build_features
from pv_anomaly.plant import PlantConfig
from pv_anomaly.simulate import simulate

PLANT = PlantConfig()
CFG = AEConfig(window=4, hidden=(32, 16), latent=4, epochs=25, patience=25, seed=3)


@pytest.fixture(scope="module")
def setup():
    """One healthy summer month: 20 days for training, 4 for validation, the rest with a 30 % power loss."""
    df, _ = simulate(PLANT, start="2021-06-01", end="2021-07-01", fault_start=None, seed=5)
    day = df.index.tz_convert(PLANT.tz).day
    train, val, test = day <= 20, (day > 20) & (day <= 24), day > 24
    faulty = df.copy()
    faulty.loc[test, "p_ac"] *= 0.7
    feats = build_features(faulty, PLANT)
    det = AutoencoderDetector(CFG)
    history = det.fit(feats, train, val)
    return det, history, feats, train, test, val


def test_training_reduces_loss(setup):
    _, history, *_ = setup
    assert len(history) == CFG.epochs
    assert history[-1]["train_loss"] < 0.5 * history[0]["train_loss"]


def test_detects_power_loss(setup):
    det, _, feats, train, test, _ = setup
    z = det.score(feats)
    day = feats["daylight"].to_numpy()
    healthy, broken = z[train & day], z[test & day]
    assert abs(healthy.median()) < 0.3
    assert broken.median() < healthy.median() - 1.0  # less power than the network reconstructs
    mse = det.score(feats, "mse")
    assert mse[test & day].median() > mse[train & day].median()


def test_deterministic_with_seed(setup):
    det, _, feats, train, _, val = setup
    again = AutoencoderDetector(CFG)
    again.fit(feats, train, val)
    pd.testing.assert_series_equal(again.score(feats), det.score(feats))


def test_save_load_roundtrip(setup, tmp_path):
    det, _, feats, *_ = setup
    path = tmp_path / "ae.pt"
    det.save(path, threshold=1.23, seed=3)
    loaded = AutoencoderDetector.load(path)
    assert loaded.config == det.config
    assert loaded.extra == {"threshold": 1.23, "seed": 3}
    for kind in ("pr_residual", "mse"):
        pd.testing.assert_series_equal(loaded.score(feats, kind), det.score(feats, kind))


def test_score_alignment_and_errors(setup):
    det, _, feats, *_ = setup
    f = feats.copy()
    f.iloc[100, 0] = np.nan
    s = det.score(f)
    assert s.iloc[:CFG.window - 1].isna().all()  # no complete window yet
    assert s.iloc[100:100 + CFG.window].isna().all() and s.iloc[100 + CFG.window:].notna().all()
    with pytest.raises(ValueError):
        det.score(feats, "nope")
