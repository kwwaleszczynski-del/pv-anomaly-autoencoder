"""NREL PVDAQ public data: download one system's daily CSV files and load them into the project's schema.

Source: Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., & Jordan, D. (2021). Photovoltaic Data
Acquisition (PVDAQ) Public Datasets. Open Energy Data Initiative (OEDI), NREL. https://doi.org/10.25984/1846021,
licensed CC BY 4.0. Files come from the public bucket at BASE_URL, one CSV per system and day.

Usage:
    from pv_anomaly.pvdaq import SYSTEMS, fetch, load
    files = fetch(34, "2016-01-01", "2017-12-31", Path("data/pvdaq"))
    df = load(files, SYSTEMS[34])          # UTC index, project column names and units
"""
from __future__ import annotations

import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pv_anomaly.plant import PlantConfig

log = logging.getLogger(__name__)

BASE_URL = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata"

# Andre Agassi Preparatory Academy, Building A, Las Vegas NV (PVDAQ system metadata): 611 Sharp 240 W mono-Si
# modules on a flat roof (tilt 11.2 deg, south), one Satcon 135 kW central inverter. Timestamps are local standard
# time without daylight saving (UTC-8).
SYSTEMS = {34: PlantConfig(latitude=36.1952, longitude=-115.1582, altitude=620.0, tz="Etc/GMT+8", tilt=11.2,
                           azimuth=180.0, pdc0_kw=146.64, pac0_kw=135.0)}
COLUMNS = {"ac_power_hw__2695": "p_ac", "poa_irradiance__2679": "poa", "module_temp_f__2689": "t_mod",
           "ambient_temp_f__2688": "t_amb", "wind_speed_mph__2691": "wind", "inverter_error_code__2690": "status"}


def day_url(system_id: int, day: pd.Timestamp) -> str:
    return (f"{BASE_URL}/system_id={system_id}/year={day.year}/month={day.month}/day={day.day}/"
            f"system_{system_id}__date_{day:%Y_%m_%d}.csv")


def fetch(system_id: int, start: str, end: str, data_dir: Path, refresh: bool = False,
          workers: int = 8) -> list[Path]:
    """Download the daily files from `start` to `end` (inclusive) into data_dir/system_<id>; cached files are kept."""
    out = Path(data_dir) / f"system_{system_id}"
    out.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    paths = [out / f"system_{system_id}__date_{d:%Y_%m_%d}.csv" for d in days]

    def get(job):
        day, path = job
        if path.exists() and not refresh:
            return False
        with urllib.request.urlopen(day_url(system_id, day), timeout=60) as r:
            path.write_bytes(r.read())
        return True

    with ThreadPoolExecutor(workers) as pool:
        new = sum(pool.map(get, zip(days, paths)))
    log.info("PVDAQ system %d: %d daily files (%d downloaded) in %s", system_id, len(paths), new, out)
    return paths


def _fahrenheit(x: pd.Series) -> pd.Series:
    """degF -> degC; exactly 0 and -40 degF are sensor dropouts at this site."""
    return ((x - 32) * 5 / 9).where((x != 0) & (x > -40))


def load(files: list[Path], plant: PlantConfig) -> pd.DataFrame:
    """Daily CSV files -> regular 15-minute UTC grid with p_ac [kW], poa [W/m2], t_mod, t_amb [degC], wind [m/s]
    and the inverter status code. Missing intervals stay NaN (the detectors skip incomplete windows)."""
    # some daily files lack channels (e.g. no inverter data during an outage): missing columns become NaN
    raw = pd.concat([pd.read_csv(f).reindex(columns=["measured_on", *COLUMNS]) for f in files], ignore_index=True)
    t = pd.to_datetime(raw["measured_on"]).dt.round("15min").dt.tz_localize(plant.tz).dt.tz_convert("UTC")
    raw = raw.rename(columns=COLUMNS).set_index(t).drop(columns="measured_on")
    raw = raw[~raw.index.duplicated()]
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="15min")
    raw = raw.reindex(grid)
    df = pd.DataFrame({
        "p_ac": (raw["p_ac"] * 0.1).clip(lower=0),  # hectowatt -> kW, night-time tare to 0
        "poa": raw["poa"].clip(lower=0),
        "t_mod": _fahrenheit(raw["t_mod"]),
        "t_amb": _fahrenheit(raw["t_amb"]),
        "wind": raw["wind"] * 0.44704,
        "status": raw["status"],
    }, index=grid)
    df.index.name = "time"
    log.info("loaded %d steps %s .. %s, %.1f %% without AC power", len(df), grid[0], grid[-1],
             100 * df["p_ac"].isna().mean())
    return df
