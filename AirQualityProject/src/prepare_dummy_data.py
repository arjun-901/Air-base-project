import pandas as pd
import numpy as np


def generate_dummy_air_quality(n_hours: int = 24 * 14) -> pd.DataFrame:
    rng = pd.date_range(start="2024-01-01", periods=n_hours, freq="H")
    rs = np.random.RandomState(42)

    pm25 = rs.normal(35, 10, size=n_hours).clip(5, 200)
    pm10 = rs.normal(60, 15, size=n_hours).clip(10, 300)
    no2 = rs.normal(20, 7, size=n_hours).clip(1, 120)
    so2 = rs.normal(8, 3, size=n_hours).clip(0.5, 50)
    co = rs.normal(0.7, 0.2, size=n_hours).clip(0.1, 5)
    o3 = rs.normal(30, 10, size=n_hours).clip(1, 200)
    temp = rs.normal(26, 4, size=n_hours).clip(-5, 45)
    humidity = rs.normal(60, 15, size=n_hours).clip(10, 100)
    wind = rs.normal(2.5, 1.0, size=n_hours).clip(0.1, 15)

    aqi = (
        0.5 * (pm25 / 12.0 * 50)
        + 0.2 * (pm10 / 20.0 * 50)
        + 0.1 * (no2 / 40.0 * 50)
        + 0.05 * (so2 / 20.0 * 50)
        + 0.05 * (co / 4.0 * 50)
        + 0.1 * (o3 / 70.0 * 50)
    )
    aqi = np.clip(aqi, 5, 500)

    df = pd.DataFrame(
        {
            "datetime": rng,
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "temp": temp,
            "humidity": humidity,
            "wind": wind,
            "AQI": aqi,
        }
    )
    return df


if __name__ == "__main__":
    out_path = "data/air_quality.csv"
    df = generate_dummy_air_quality()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


