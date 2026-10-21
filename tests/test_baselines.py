import numpy as np
import pandas as pd
import pytest

from pv_anomaly.baselines import PhysicalBaseline, PRBaseline, RegressionBaseline
from pv_anomaly.plant import PlantConfig

PLANT = PlantConfig()


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 4000
    poa = rng.uniform(60, 1000, n)
    df = pd.DataFrame({"poa": poa, "t_amb": rng.uniform(-5, 30, n), "wind": rng.uniform(0.5, 8, n)})
    df["t_mod"] = df["t_amb"] + 0.03 * poa
    return df


def test_regression_recovers_coefficients(data):
    coef = np.array([0.85, -1.2e-4, -3e-3, 4e-3])
    data["p_ac"] = data["poa"] * (coef[0] + coef[1] * data["poa"] + coef[2] * data["t_amb"] + coef[3] * data["wind"])
    data["p_ac"] *= 1 + 0.01 * np.random.default_rng(1).standard_normal(len(data))
    model = RegressionBaseline(PLANT).fit(data, np.ones(len(data), bool))
    assert model.coef == pytest.approx(coef, rel=0.05)
    z = model.residual_z(data)
    assert abs(z.median()) < 0.1
    faulty = data.assign(p_ac=0.85 * data["p_ac"])
    assert model.residual_z(faulty).median() < -3


def test_physical_learns_scale_factor(data):
    raw = PhysicalBaseline(PLANT)
    data["p_ac"] = 0.9 * raw.expected(data)  # plant performs 10 % below the nameplate model
    model = PhysicalBaseline(PLANT).fit(data, np.ones(len(data), bool))
    assert model.k == pytest.approx(0.9, rel=0.01)
    assert model.residual_z(data).abs().median() < 1


def test_pr_baseline_flags_drop_and_ignores_dark(data):
    noise = 1 + 0.02 * np.random.default_rng(2).standard_normal(len(data))
    data["p_ac"] = 0.8 * PLANT.pdc0_kw * data["poa"] / 1000 * noise
    model = PRBaseline(PLANT).fit(data, np.ones(len(data), bool))
    assert model.residual_z(data.assign(p_ac=0.5 * data["p_ac"])).median() < -5
    dark = data.assign(poa=10.0)
    assert model.residual_z(dark).isna().all()


def test_regression_without_wind_on_module_temperature(data):
    data["p_ac"] = data["poa"] * (0.85 - 1e-4 * data["poa"] - 2e-3 * data["t_mod"])
    model = RegressionBaseline(PLANT, temp="t_mod", wind=False).fit(data.drop(columns="wind"), np.ones(len(data), bool))
    assert model.coef == pytest.approx([0.85, -1e-4, -2e-3], rel=1e-6)
