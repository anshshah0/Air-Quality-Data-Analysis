import argparse
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from pathlib import Path

from .utils import standardize_columns, coerce_date

def build_industrial_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Industrial activity proxy:
    - z-score NO2, SO2, CO (if present)
    - sum them and take 7-day rolling mean per city
    """
    df = df.copy()
    for col in ["no2", "so2", "co"]:
        if col in df.columns:
            mu = df[col].mean()
            sd = df[col].std(ddof=0) or 1.0
            df[f"{col}_z"] = (df[col] - mu) / sd

    z_cols = [c for c in ["no2_z", "so2_z", "co_z"] if c in df.columns]
    if z_cols:
        df["industrial_activity_proxy_raw"] = df[z_cols].sum(axis=1)
        if "city" in df.columns:
            df = df.sort_values(["city", "date"])
            df["industrial_activity_proxy"] = (
                df.groupby("city")["industrial_activity_proxy_raw"]
                  .transform(lambda s: s.rolling(window=7, min_periods=3).mean())
            )
        else:
            df = df.sort_values("date")
            df["industrial_activity_proxy"] = df["industrial_activity_proxy_raw"].rolling(
                window=7, min_periods=3
            ).mean()
    else:
        df["industrial_activity_proxy"] = np.nan
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = coerce_date(df, "date")
    keep = ["date", "city", "aqi", "pm25", "pm10", "no2", "so2", "co", "o3"]
    exist = [c for c in keep if c in df.columns]
    df = df[exist + [c for c in df.columns if c not in exist]]

    df = df.dropna(subset=["date"])
    if "aqi" in df.columns:
        df = df[df["aqi"].between(0, 500, inclusive="both") | df["aqi"].isna()]

    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df = df.drop_duplicates()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    df = build_industrial_proxy(df)

    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if col in df.columns:
            hi = df[col].quantile(0.999)
            df = df[df[col] <= hi]

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output Parquet")
    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    df = clean(df)

    n_rows = len(df)
    n_cities = df["city"].nunique() if "city" in df.columns else None
    print(f"[clean] rows={n_rows}, cities={n_cities}")

    df.to_parquet(outp, index=False)
    print(f"[clean] wrote -> {outp}")

if __name__ == "__main__":
    main()
