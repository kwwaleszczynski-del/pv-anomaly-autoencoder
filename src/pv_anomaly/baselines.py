"""Reference detectors: residuals of simple performance models fitted on healthy data.

    PRBaseline          instantaneous performance ratio vs. its training median
    RegressionBaseline  ASTM E2848-style regression P = POA * (a1 + a2*POA + a3*T_amb + a4*wind); the
                        temperature column and the wind term can be changed when a sensor is unreliable
    PhysicalBaseline    pvlib PVWatts with nameplate data, scaled by one factor fitted on healthy data

A residual is (measured - expected) divided either by the DC nameplate (``units="kw"``: weights points by power,
so noisy low-light periods count little) or by DC nameplate * POA / 1000 (``units="pr"``: a performance-ratio
deviation, the unit the autoencoder works in). `residual_z` divides by the robust training spread
(1.4826 * MAD); negative means less power than expected.

Usage:
    from pv_anomaly.baselines import RegressionBaseline
    model = RegressionBaseline(plant).fit(df, train_mask)
    z = model.residual_z(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from pv_anomaly.features import LOW_POA
from pv_anomaly.plant import PlantConfig, inverter_ac


def _robust_scale(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


class _Baseline:
    name = "baseline"

    units = "pr"

    def __init__(self, plant: PlantConfig, units: str | None = None):
        self.plant = plant
        self.units = units or self.units
        self.center, self.scale = 0.0, 1.0

    def _usable(self, df: pd.DataFrame, mask) -> np.ndarray:
        """Training points: selected by `mask`, enough irradiance, inverter not clipping."""
        bright = (df["poa"] >= LOW_POA).to_numpy()
        return np.asarray(mask) & bright & (df["p_ac"] < 0.98 * self.plant.pac0_kw).to_numpy()

    def expected(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def residual(self, df: pd.DataFrame) -> pd.Series:
        ref = self.plant.pdc0_kw * (df["poa"] / 1000 if self.units == "pr" else 1.0)
        return ((df["p_ac"] - self.expected(df)) / ref).where(df["poa"] >= LOW_POA)

    def fit(self, df: pd.DataFrame, mask) -> "_Baseline":
        r = self.residual(df).to_numpy()[self._usable(df, mask)]
        self.center, self.scale = float(np.nanmedian(r)), _robust_scale(r)
        return self

    def residual_z(self, df: pd.DataFrame) -> pd.Series:
        return (self.residual(df) - self.center) / self.scale


class PRBaseline(_Baseline):
    name = "pr_threshold"

    def expected(self, df):
        return 0.0 * df["p_ac"]  # the residual is the PR itself; fit() subtracts its training median


class RegressionBaseline(_Baseline):
    name = "regression"
    units = "kw"  # chosen on the development seeds; the physical model is better in "pr"

    def __init__(self, plant: PlantConfig, units: str | None = None, temp: str = "t_amb", wind: bool = True):
        super().__init__(plant, units)
        self.temp, self.wind = temp, wind
        self.coef = np.zeros(4 if wind else 3)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        poa = df["poa"].to_numpy()
        cols = [poa, poa**2, poa * df[self.temp].to_numpy()]
        return np.column_stack(cols + [poa * df["wind"].to_numpy()] if self.wind else cols)

    def fit(self, df, mask):
        use = self._usable(df, mask)
        A, y = self._design(df)[use], df["p_ac"].to_numpy()[use]
        ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
        self.coef = np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
        return super().fit(df, mask)

    def expected(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.clip(self._design(df) @ self.coef, 0, self.plant.pac0_kw), index=df.index)


class PhysicalBaseline(_Baseline):
    name = "physical"
    GAMMA = -0.004  # datasheet-style temperature coefficient [1/K]

    def __init__(self, plant: PlantConfig, units: str | None = None):
        super().__init__(plant, units)
        self.k = 1.0

    def _pdc(self, df: pd.DataFrame) -> np.ndarray:
        t_cell = df["t_mod"] + 3.0 * df["poa"] / 1000  # SAPM back-of-module to cell offset
        return np.asarray(pvlib.pvsystem.pvwatts_dc(df["poa"], t_cell, self.plant.pdc0_kw, self.GAMMA), float)

    def fit(self, df, mask):
        use = self._usable(df, mask) & (df["poa"] >= 200).to_numpy()
        ratio = df["p_ac"].to_numpy()[use] / inverter_ac(self._pdc(df)[use], self.plant)
        self.k = float(np.nanmedian(ratio))
        return super().fit(df, mask)

    def expected(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(inverter_ac(self.k * self._pdc(df), self.plant), index=df.index)


BASELINES = (PRBaseline, RegressionBaseline, PhysicalBaseline)
