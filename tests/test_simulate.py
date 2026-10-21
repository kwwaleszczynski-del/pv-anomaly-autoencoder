import numpy as np
import pandas as pd
import pytest

from pv_anomaly.plant import PlantConfig, solar_geometry
from pv_anomaly.simulate import FAULT_TYPES, FaultEvent, apply_faults, schedule_faults, simulate

PLANT = PlantConfig()


@pytest.fixture(scope="module")
def sim():
    return simulate(PLANT, start="2021-05-01", end="2021-08-01", fault_start="2021-06-01", per_type=1, seed=1)


def test_reproducible_with_seed(sim):
    df, events = simulate(PLANT, start="2021-05-01", end="2021-08-01", fault_start="2021-06-01", per_type=1, seed=1)
    pd.testing.assert_frame_equal(df, sim[0])
    pd.testing.assert_frame_equal(events, sim[1])
    other, _ = simulate(PLANT, start="2021-05-01", end="2021-08-01", fault_start="2021-06-01", per_type=1, seed=2)
    assert not np.allclose(other["p_ac"], df["p_ac"])


def test_regular_grid_and_no_gaps(sim):
    df, _ = sim
    assert df.index.tz is not None
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()  # resolution-independent (ns/us)
    assert not df[["p_ac", "poa", "t_mod", "t_amb", "wind"]].isna().any().any()


def test_physics_sanity(sim):
    df, _ = sim
    geo = solar_geometry(df.index, PLANT)
    night = geo["elevation"] < -5
    assert (df.loc[night, "p_ac"] == 0).all()
    assert df["p_ac_healthy"].max() <= PLANT.pac0_kw + 1e-9
    # healthy summer performance ratio around noon on clear days
    healthy = ((df["fault"] == "") & (df["sky"] == "clear") & (geo["elevation"] > 45)
               & (df["p_ac"] < 0.95 * PLANT.pac0_kw))
    pr = df.loc[healthy, "p_ac"] / (PLANT.pdc0_kw * df.loc[healthy, "poa"] / 1000)
    assert 0.75 < pr.median() < 0.92
    # module hotter than air in the sun
    assert (df.loc[geo["elevation"] > 40, "t_mod"] - df.loc[geo["elevation"] > 40, "t_amb"]).median() > 5


def test_schedule_is_sorted_and_non_overlapping():
    rng = np.random.default_rng(0)
    start, end = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")
    events = schedule_faults(start, end, PLANT, rng, per_type=4)
    assert len(events) == 20
    assert sorted(e.kind for e in events) == sorted(FAULT_TYPES * 4)
    for a, b in zip(events, events[1:]):
        assert a.end < b.start
    assert events[0].start >= start and events[-1].end <= end


@pytest.fixture(scope="module")
def day():
    index = pd.date_range("2021-06-10", periods=96 * 3, freq="15min", tz="UTC")
    return index, solar_geometry(index, PLANT)


def _event(kind, index, severity, **params):
    return FaultEvent(0, kind, index[96], index[192], severity, params)


def test_string_outage_and_soiling(day):
    index, geo = day
    f = apply_faults(index, geo, np.ones(len(index)), PLANT, [_event("string_outage", index, 2 / 16)])
    assert np.allclose(f["dc_mult"][96:192], 1 - 2 / 16) and np.allclose(f["dc_mult"][:96], 1)
    assert (f["fault"][96:192] == "string_outage").all() and (f["fault"][192:] == "").all()
    f = apply_faults(index, geo, np.ones(len(index)), PLANT, [_event("soiling", index, 0.1)])
    loss = 1 - f["dc_mult"][96:192]
    assert loss[0] == 0 and np.all(np.diff(loss) > 0) and loss[-1] < 0.1


def test_derate_drift_and_shading(day):
    index, geo = day
    f = apply_faults(index, geo, np.ones(len(index)), PLANT, [_event("inverter_derate", index, 0.3)])
    assert np.allclose(f["ac_limit"][96:192], 0.7 * PLANT.pac0_kw) and np.isinf(f["ac_limit"][:96]).all()
    f = apply_faults(index, geo, np.ones(len(index)), PLANT, [_event("sensor_drift", index, -0.1)])
    assert np.allclose(f["dc_mult"], 1) and f["sensor_mult"][191] < 0.91
    ev = _event("shading", index, 0.5, azimuth=90.0, max_elevation=30.0, half_width=20.0)
    f = apply_faults(index, geo, np.full(len(index), 0.8), PLANT, [ev])
    shaded = f["dc_mult"] < 1
    assert shaded.any()
    assert np.allclose(f["dc_mult"][shaded], 1 - 0.5 * 0.8)
    assert (np.abs(geo["azimuth"].to_numpy()[shaded] - 90) < 20).all()
    assert (geo["elevation"].to_numpy()[shaded] < 30).all()


def test_fault_effects_in_simulation(sim):
    df, events = sim
    for ev in events.itertuples():
        m = (df.index >= ev.start) & (df.index < ev.end) & (df["p_ac_healthy"] > 100)
        ratio = (df.loc[m, "p_ac"] / df.loc[m, "p_ac_healthy"]).median()
        if ev.kind == "string_outage":
            assert ratio == pytest.approx(1 - ev.severity, abs=0.02)
        elif ev.kind == "sensor_drift":
            assert ratio == pytest.approx(1, abs=0.01)
        elif ev.kind == "inverter_derate":
            assert df.loc[m, "p_ac"].max() <= (1 - ev.severity) * PLANT.pac0_kw * 1.03
    assert (df.loc[df["fault"] == "", "impact"] == 0).all()
