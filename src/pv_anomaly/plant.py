"""PV plant definition, solar geometry and the 'true' physical model used by the simulator.

The simulator deliberately uses a richer model (incidence-angle losses, ADR efficiency with low-light and
temperature effects, SAPM module temperature, hidden DC losses) than the detectors are given, so that a
nameplate model is never a perfect fit - just like on a real plant.

Usage:
    from pv_anomaly.plant import PlantConfig, solar_geometry, plant_power
    plant = PlantConfig()
    geo = solar_geometry(index, plant)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

# glass/polymer modules on an open rack
SAPM = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
# ADR relative-efficiency coefficients of a generic c-Si module (pvlib ADR model example)
ADR = {"k_a": 0.99924, "k_d": -5.49097, "tc_d": 0.01918, "k_rs": 0.06999, "k_rsh": 0.26144}
INV_ETA = 0.96


@dataclass(frozen=True)
class PlantConfig:
    """A generic ground-mounted plant: 1 MWp DC, 800 kW inverter (DC/AC 1.25), 16 strings, fixed tilt, south."""

    latitude: float = 52.2
    longitude: float = 21.0
    altitude: float = 100.0
    tz: str = "Europe/Warsaw"
    tilt: float = 30.0
    azimuth: float = 180.0
    pdc0_kw: float = 1000.0
    pac0_kw: float = 800.0
    n_strings: int = 16
    dc_loss: float = 0.03  # wiring + mismatch; not known to the detectors

    @property
    def location(self) -> pvlib.location.Location:
        return pvlib.location.Location(self.latitude, self.longitude, altitude=self.altitude)


def solar_geometry(index: pd.DatetimeIndex, plant: PlantConfig) -> pd.DataFrame:
    """Sun position, angle of incidence and clear-sky irradiance (horizontal and in the module plane)."""
    loc = plant.location
    sp = loc.get_solarposition(index)
    cs = loc.get_clearsky(index, solar_position=sp)
    aoi = pvlib.irradiance.aoi(plant.tilt, plant.azimuth, sp["apparent_zenith"], sp["azimuth"])
    poa_cs = pvlib.irradiance.get_total_irradiance(plant.tilt, plant.azimuth, sp["apparent_zenith"], sp["azimuth"],
                                                   cs["dni"], cs["ghi"], cs["dhi"])["poa_global"]
    return pd.DataFrame({
        "elevation": sp["apparent_elevation"], "azimuth": sp["azimuth"], "zenith": sp["apparent_zenith"],
        "aoi": aoi, "ghi_clearsky": cs["ghi"], "dni_clearsky": cs["dni"], "poa_clearsky": poa_cs.fillna(0.0),
    }, index=index)


def plant_power(poa_direct, poa_diffuse, aoi, temp_air, wind_speed, plant: PlantConfig) -> dict[str, np.ndarray]:
    """'True' plant model: effective irradiance, module/cell temperature and DC power [kW].

    Returns a dict with ``eff_irr``, ``beam_frac`` (share of beam in effective irradiance), ``t_mod``, ``pdc``.
    """
    poa_direct = np.asarray(poa_direct, float)
    poa_diffuse = np.asarray(poa_diffuse, float)
    poa_global = poa_direct + poa_diffuse
    beam = poa_direct * pvlib.iam.physical(np.asarray(aoi, float))
    eff_irr = beam + 0.95 * poa_diffuse
    t_mod = pvlib.temperature.sapm_module(poa_global, np.asarray(temp_air, float), np.asarray(wind_speed, float),
                                          SAPM["a"], SAPM["b"])
    t_cell = pvlib.temperature.sapm_cell_from_module(t_mod, poa_global, SAPM["deltaT"])
    eta = pvlib.pvarray.pvefficiency_adr(eff_irr, t_cell, **ADR)
    pdc = np.where(eff_irr > 0, plant.pdc0_kw * eff_irr / 1000.0 * eta * (1 - plant.dc_loss), 0.0)
    beam_frac = np.divide(beam, eff_irr, out=np.zeros_like(eff_irr), where=eff_irr > 0)
    return {"eff_irr": eff_irr, "beam_frac": beam_frac, "t_mod": np.asarray(t_mod, float), "pdc": np.maximum(pdc, 0.0)}


def inverter_ac(pdc, plant: PlantConfig) -> np.ndarray:
    """PVWatts inverter: part-load efficiency curve and clipping at the AC rating."""
    return np.asarray(pvlib.inverter.pvwatts(np.asarray(pdc, float), plant.pac0_kw / INV_ETA, eta_inv_nom=INV_ETA))
