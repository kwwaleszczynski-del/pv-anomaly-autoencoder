"""Synthetic 15-minute data of a PV plant with injected, labelled faults.

Weather comes from a stochastic model on top of the pvlib clear sky (no downloads): every day gets a sky regime
(clear / broken clouds / overcast) from a seasonal Markov chain, the clear-sky index follows an AR(1) process
within the day, and ambient temperature and wind follow seasonal and diurnal cycles with slow random anomalies.

Five fault types are injected in the evaluation period, one at a time with healthy gaps between them:

    soiling          gradual DC loss growing linearly to `severity`, removed by cleaning at the end
    shading          a new obstruction (tree, building) blocks `severity` of the beam irradiance when the low
                     sun is behind it - mostly a winter effect
    string_outage    `severity` = k / n_strings of the DC power lost (blown fuse, open connector)
    inverter_derate  AC power capped at (1 - severity) of the rating (overheating, wrong set-point)
    sensor_drift     the plane-of-array pyranometer drifts linearly to (1 + severity), plant itself is healthy

Usage:
    from pv_anomaly.simulate import simulate
    df, events = simulate(start="2021-01-01", end="2024-01-01", fault_start="2022-01-01", seed=42)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pvlib

from pv_anomaly.plant import PlantConfig, inverter_ac, plant_power, solar_geometry

FREQ = "15min"
FAULT_TYPES = ("soiling", "shading", "string_outage", "inverter_derate", "sensor_drift")
# sky regimes: mean clear-sky index, intraday standard deviation, AR(1) coefficient per 15 min
REGIMES = (("clear", 0.97, 0.03, 0.90), ("broken", 0.62, 0.25, 0.60), ("overcast", 0.28, 0.08, 0.90))
DURATION_DAYS = {"soiling": (15, 35), "shading": (7, 21), "string_outage": (2, 8),
                 "inverter_derate": (3, 10), "sensor_drift": (10, 30)}


@dataclass
class FaultEvent:
    fault_id: int
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    severity: float
    params: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"fault_id": self.fault_id, "kind": self.kind, "start": self.start, "end": self.end,
                "days": round((self.end - self.start) / pd.Timedelta("1D"), 2), "severity": round(self.severity, 4),
                "params": json.dumps(self.params)}


def _ar1(phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AR(1) process with unit stationary variance and a time-varying coefficient."""
    e = rng.standard_normal(len(phi)) * np.sqrt(1 - phi**2)
    x = np.empty(len(phi))
    x[0] = rng.standard_normal()
    for i in range(1, len(phi)):
        x[i] = phi[i] * x[i - 1] + e[i]
    return x


def _daily_regimes(days: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Sky regime per day: Markov chain that keeps yesterday's regime with p=0.5, else draws a seasonal one."""
    s = np.cos(2 * np.pi * (days.dayofyear.values - 172) / 365.25)  # +1 midsummer, -1 midwinter
    p = np.stack([0.25 + 0.10 * s, 0.40 + 0.08 * s, 0.35 - 0.18 * s], axis=1)
    p /= p.sum(axis=1, keepdims=True)
    out = np.empty(len(days), dtype=int)
    out[0] = rng.choice(3, p=p[0])
    for i in range(1, len(days)):
        out[i] = out[i - 1] if rng.random() < 0.5 else rng.choice(3, p=p[i])
    return out


def simulate_weather(index: pd.DatetimeIndex, geo: pd.DataFrame, plant: PlantConfig,
                     rng: np.random.Generator) -> pd.DataFrame:
    """GHI/DNI/DHI, plane-of-array components, ambient temperature, wind speed and the sky regime."""
    local = index.tz_convert(plant.tz)
    day = local.normalize().tz_localize(None)
    days = pd.DatetimeIndex(day.unique())
    regime_day = pd.Series(_daily_regimes(days, rng), index=days)
    regime = regime_day.reindex(day).to_numpy()
    mean, sd, phi = (np.array([r[k] for r in REGIMES])[regime] for k in (1, 2, 3))

    kc = np.clip(mean + sd * _ar1(phi, rng), 0.03, 1.15)
    ghi = kc * geo["ghi_clearsky"].to_numpy()
    erbs = pvlib.irradiance.erbs(ghi, geo["zenith"].to_numpy(), index)
    dni = np.clip(np.nan_to_num(erbs["dni"]), 0, 1.1 * geo["dni_clearsky"].to_numpy())
    cosz = np.cos(np.radians(geo["zenith"].to_numpy()))
    dhi = np.clip(ghi - dni * np.clip(cosz, 0, None), 0, None)
    poa = pvlib.irradiance.get_total_irradiance(plant.tilt, plant.azimuth, geo["zenith"], geo["azimuth"],
                                                dni, ghi, dhi)

    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    doy = local.dayofyear.to_numpy()
    day_kc = pd.Series(mean, index=index).groupby(day).transform("mean").to_numpy()
    t_season = 8.5 + 10.5 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    t_diurnal = (2.0 + 6.0 * day_kc) * np.cos(2 * np.pi * (hour - 15) / 24)
    temp_air = t_season + t_diurnal + 3.0 * _ar1(np.full(len(index), 0.999), rng)

    wind_day = pd.Series(rng.gamma(2.0, 1.7, len(days)), index=days).reindex(day).to_numpy()
    wind = np.clip(wind_day * (1 + 0.3 * np.sin(np.pi * np.clip((hour - 6) / 12, 0, 1)))
                   + 0.8 * _ar1(np.full(len(index), 0.9), rng), 0.3, None)

    return pd.DataFrame({
        "ghi": ghi, "dni": dni, "dhi": dhi,
        "poa_direct": poa["poa_direct"].fillna(0).clip(lower=0).to_numpy(),
        "poa_diffuse": poa["poa_diffuse"].fillna(0).clip(lower=0).to_numpy(),
        "temp_air": temp_air, "wind": wind, "sky": np.array([r[0] for r in REGIMES])[regime],
    }, index=index)


def schedule_faults(start: pd.Timestamp, end: pd.Timestamp, plant: PlantConfig, rng: np.random.Generator,
                    per_type: int = 4) -> list[FaultEvent]:
    """Non-overlapping fault events between `start` and `end`: `per_type` rounds of all five types in random
    order, with 5-25 healthy days between events. Events that would not fit before `end` are dropped."""
    events, t = [], start + pd.Timedelta(days=float(rng.uniform(5, 25)))
    for _ in range(per_type):
        for kind in rng.permutation(FAULT_TYPES):
            lo, hi = DURATION_DAYS[kind]
            t0 = t.floor("1D") + pd.Timedelta(minutes=15 * int(rng.integers(0, 96)))
            t1 = (t0 + pd.Timedelta(days=float(rng.uniform(lo, hi)))).round(FREQ)
            if t1 > end:
                return events
            params = {}
            if kind == "soiling":
                sev = rng.uniform(0.04, 0.15)
            elif kind == "shading":
                sev = rng.uniform(0.4, 0.8)
                # an obstruction south-east or south-west of the array: shades the low winter sun
                east = rng.random() < 0.5
                params = {"azimuth": round(float(rng.uniform(130, 160) if east else rng.uniform(200, 230)), 1),
                          "max_elevation": round(float(rng.uniform(15, 30)), 1), "half_width": 20.0}
            elif kind == "string_outage":
                k = int(rng.integers(1, 4))
                sev, params = k / plant.n_strings, {"strings_off": k}
            elif kind == "inverter_derate":
                sev = rng.uniform(0.2, 0.45)
            else:
                sev = rng.uniform(0.06, 0.15) * rng.choice([-1, 1])
            events.append(FaultEvent(len(events), str(kind), t0, t1, float(sev), params))
            t = t1 + pd.Timedelta(days=float(rng.uniform(5, 25)))
    return events


def apply_faults(index: pd.DatetimeIndex, geo: pd.DataFrame, beam_frac: np.ndarray, plant: PlantConfig,
                 events: list[FaultEvent]) -> dict[str, np.ndarray]:
    """Multipliers on DC power and on the irradiance sensor, AC limit, and the fault kind/id per timestamp."""
    n = len(index)
    dc_mult, sensor_mult = np.ones(n), np.ones(n)
    ac_limit = np.full(n, np.inf)
    kind = np.full(n, "", dtype=object)
    fid = np.full(n, -1)
    for ev in events:
        m = (index >= ev.start) & (index < ev.end)
        progress = ((index[m] - ev.start) / (ev.end - ev.start)).to_numpy(dtype=float)
        if ev.kind == "soiling":
            dc_mult[m] *= 1 - ev.severity * progress
        elif ev.kind == "shading":
            p = ev.params
            daz = np.abs(geo["azimuth"].to_numpy()[m] - p["azimuth"])
            behind = (daz < p["half_width"]) & (geo["elevation"].to_numpy()[m] < p["max_elevation"])
            dc_mult[m] *= np.where(behind, 1 - ev.severity * beam_frac[m], 1.0)
        elif ev.kind == "string_outage":
            dc_mult[m] *= 1 - ev.severity
        elif ev.kind == "inverter_derate":
            ac_limit[m] = (1 - ev.severity) * plant.pac0_kw
        elif ev.kind == "sensor_drift":
            sensor_mult[m] *= 1 + ev.severity * progress
        kind[m], fid[m] = ev.kind, ev.fault_id
    return {"dc_mult": dc_mult, "sensor_mult": sensor_mult, "ac_limit": ac_limit, "fault": kind, "fault_id": fid}


def simulate(plant: PlantConfig = PlantConfig(), start: str = "2021-01-01", end: str = "2024-01-01",
             fault_start: str | None = "2022-01-01", per_type: int = 4, seed: int = 42,
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate 15-minute data from `start` to `end` (local dates, UTC index) with faults from `fault_start` on.

    Returns ``(data, events)``. Measured columns: ``p_ac`` [kW], ``poa`` [W/m2], ``t_mod``, ``t_amb`` [degC],
    ``wind`` [m/s]. Ground truth (never shown to the detectors): ``p_ac_healthy`` (noise-free AC power without the
    fault), ``fault``, ``fault_id``, ``impact`` (relative effect of the fault at that time) and ``sky``.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(pd.Timestamp(start, tz=plant.tz), pd.Timestamp(end, tz=plant.tz), freq=FREQ,
                          inclusive="left").tz_convert("UTC")
    geo = solar_geometry(index, plant)
    w = simulate_weather(index, geo, plant, rng)
    model = plant_power(w["poa_direct"], w["poa_diffuse"], geo["aoi"], w["temp_air"], w["wind"], plant)

    events = []
    if fault_start is not None:
        events = schedule_faults(pd.Timestamp(fault_start, tz=plant.tz).tz_convert("UTC"), index[-1], plant, rng,
                                 per_type)
    f = apply_faults(index, geo, model["beam_frac"], plant, events)

    p_healthy = inverter_ac(model["pdc"], plant)
    p_fault = np.minimum(inverter_ac(model["pdc"] * f["dc_mult"], plant), f["ac_limit"])
    power_impact = np.divide(p_healthy - p_fault, p_healthy, out=np.zeros(len(index)), where=p_healthy > 1.0)
    impact = np.maximum(power_impact, np.abs(f["sensor_mult"] - 1))

    poa_true = w["poa_direct"].to_numpy() + w["poa_diffuse"].to_numpy()
    # point pyranometer vs. array average: larger mismatch under broken clouds
    poa_noise = np.where(w["sky"].to_numpy() == "broken", 0.05, 0.01)
    data = pd.DataFrame({
        "p_ac": np.clip(p_fault * (1 + 0.005 * rng.standard_normal(len(index))), 0, None),
        "poa": np.clip(poa_true * f["sensor_mult"] * (1 + poa_noise * rng.standard_normal(len(index))), 0, None),
        "t_mod": model["t_mod"] + 0.5 * rng.standard_normal(len(index)),
        "t_amb": w["temp_air"].to_numpy() + 0.3 * rng.standard_normal(len(index)),
        "wind": w["wind"].to_numpy(),
        "p_ac_healthy": p_healthy,
        "fault": f["fault"].astype(str),
        "fault_id": f["fault_id"],
        "impact": impact,
        "sky": w["sky"].to_numpy(),
    }, index=index)
    data.index.name = "time"
    events_df = pd.DataFrame([e.as_row() for e in events],
                             columns=["fault_id", "kind", "start", "end", "days", "severity", "params"])
    return data, events_df
