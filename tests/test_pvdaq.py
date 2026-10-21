import numpy as np
import pandas as pd
import pytest

from pv_anomaly.pvdaq import SYSTEMS, day_url, load

HEADER = ("measured_on,ac_current__2685,ac_power_hw__2695,ambient_temp_f__2688,inverter_error_code__2690,"
          "module_temp_f__2689,poa_irradiance__2679,wind_speed_mph__2691,system_id")


def test_day_url():
    url = day_url(34, pd.Timestamp("2016-03-05"))
    assert url.endswith("/system_id=34/year=2016/month=3/day=5/system_34__date_2016_03_05.csv")
    assert url.startswith("https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata")


def test_load_units_time_zone_and_gaps(tmp_path):
    day1 = tmp_path / "system_34__date_2016_06_01.csv"
    day1.write_text("\n".join([
        HEADER,
        "2016-06-01 00:00:00,0,-2.0,77.0,0,68.0,0.3,2.0,34",
        "2016-06-01 00:15:01,0,500.0,212.0,0,0.0,800.0,10.0,34",  # 0 degF module temperature is a dropout
        "2016-06-01 00:29:59,0,1000.0,-40.0,0,104.0,900.0,0.0,34",  # -40 degF ambient is a dropout
    ]))
    day2 = tmp_path / "system_34__date_2016_06_02.csv"  # outage file without the inverter channels
    day2.write_text("measured_on,inverter_error_code__2690,module_temp_f__2689,poa_irradiance__2679,"
                    "wind_speed_mph__2691,system_id\n2016-06-01 01:00:00,139,50.0,0.0,1.0,34\n")
    df = load([day1, day2], SYSTEMS[34])
    assert df.index[0] == pd.Timestamp("2016-06-01 08:00", tz="UTC")  # local standard time UTC-8
    assert len(df) == 5 and (df.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()
    assert df["p_ac"].iloc[:3].tolist() == [0.0, 50.0, 100.0]  # hectowatt -> kW, negative tare clipped
    assert df["t_mod"].iloc[0] == pytest.approx(20.0) and np.isnan(df["t_mod"].iloc[1])
    assert df["t_amb"].iloc[1] == pytest.approx(100.0) and np.isnan(df["t_amb"].iloc[2])
    assert df["wind"].iloc[1] == pytest.approx(4.4704)
    assert df.iloc[3].isna().all()  # 00:45 missing in the source
    assert np.isnan(df["p_ac"].iloc[4]) and df["status"].iloc[4] == 139
