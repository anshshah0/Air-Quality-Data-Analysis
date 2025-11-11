import re
import pandas as pd

POLLUTANT_ALIASES = {
    "pm25": ["pm25", "pm_2_5", "pm2_5", "pm2.5", "pm-2.5"],
    "pm10": ["pm10", "pm_10", "pm-10"],
    "no2":  ["no2", "nitrogen_dioxide"],
    "so2":  ["so2", "sulfur_dioxide", "sulphur_dioxide"],
    "co":   ["co", "carbon_monoxide"],
    "o3":   ["o3", "ozone"],
    "aqi":  ["aqi", "AQI", "AirQualityIndex", "air_quality_index"]
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c):
        c = c.strip().lower()
        c = re.sub(r'[^a-z0-9_]+', '_', c)
        return c
    df = df.rename(columns={c: norm(c) for c in df.columns})
    col_map = {}
    for std, aliases in POLLUTANT_ALIASES.items():
        for a in aliases:
            if a.lower() in df.columns:
                col_map[a.lower()] = std
                break
    df = df.rename(columns=col_map)
    return df

def coerce_date(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    if date_col not in df.columns:
        candidates = [c for c in df.columns if "date" in c]
        if candidates:
            date_col = candidates[0]
        else:
            raise ValueError("No 'date' column found; please ensure data has a date field.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    return df.rename(columns={date_col: "date"})
