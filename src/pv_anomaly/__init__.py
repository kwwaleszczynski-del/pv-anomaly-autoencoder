"""Fault detection for PV plants: autoencoder vs. statistical and physical baselines on simulated data."""

from pv_anomaly.plant import PlantConfig
from pv_anomaly.simulate import FAULT_TYPES, simulate

__all__ = ["FAULT_TYPES", "PlantConfig", "simulate"]
__version__ = "0.1.0"
