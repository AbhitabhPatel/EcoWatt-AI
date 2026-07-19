import numpy as np
import pandas as pd


def create_features(df):

    df = df.copy()

    # Ensure datetime column exists
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    # ==========================
    # Cyclical Features
    # ==========================

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dayofyear"] = df["datetime"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["year_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["year_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # ==========================
    # Lag Features
    # ==========================

    for lag in [1, 5, 15, 60]:
        df[f"active_power_lag_{lag}"] = (
            df["Global_active_power"].shift(lag)
        )

    # ==========================
    # Rolling Features
    # ==========================

    for window in [15, 60]:

        rolling = df["Global_active_power"].rolling(window)

        df[f"active_power_rolling_mean_{window}"] = rolling.mean()
        df[f"active_power_rolling_std_{window}"] = rolling.std()
        df[f"active_power_rolling_min_{window}"] = rolling.min()
        df[f"active_power_rolling_max_{window}"] = rolling.max()

    # ==========================
    # Behaviour Features
    # ==========================

    df["active_power_change_1"] = (
        df["Global_active_power"].diff()
    )

    df["deviation_from_15min_mean"] = (
        df["Global_active_power"]
        - df["active_power_rolling_mean_15"]
    )

    df["deviation_from_60min_mean"] = (
        df["Global_active_power"]
        - df["active_power_rolling_mean_60"]
    )

    df["rolling_zscore_15"] = (
        (
            df["Global_active_power"]
            - df["active_power_rolling_mean_15"]
        )
        /
        (df["active_power_rolling_std_15"] + 1e-6)
    )

    df["rolling_zscore_60"] = (
        (
            df["Global_active_power"]
            - df["active_power_rolling_mean_60"]
        )
        /
        (df["active_power_rolling_std_60"] + 1e-6)
    )

    # ==========================
    # Log Features
    # ==========================

    df["active_power_change_rate_log"] = np.log1p(
        np.abs(df["active_power_change_1"])
    )

    df["rolling_zscore_15_log"] = np.log1p(
        np.abs(df["rolling_zscore_15"])
    )

    df["rolling_zscore_60_log"] = np.log1p(
        np.abs(df["rolling_zscore_60"])
    )

    # Remove NaNs introduced by lag/rolling operations
    df = df.dropna().reset_index(drop=True)

    return df