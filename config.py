
# ===========================
# Dataset Paths
# ===========================

DEFAULT_DATASET = "data/processed/dashboard_dataset.csv"

MODEL_PATH = "models/best_isolation_forest.pkl"

SCALER_PATH = "models/standard_scaler.pkl"

# ===========================
# Required Columns
# ===========================

required_columns = [
    "datetime",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

# ===========================
# Feature Columns
# ===========================

feature_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "year_sin",
    "year_cos",

    "active_power_lag_1",
    "active_power_lag_5",
    "active_power_lag_15",
    "active_power_lag_60",

    "active_power_rolling_mean_15",
    "active_power_rolling_std_15",
    "active_power_rolling_min_15",
    "active_power_rolling_max_15",

    "active_power_rolling_mean_60",
    "active_power_rolling_std_60",
    "active_power_rolling_min_60",
    "active_power_rolling_max_60",

    "active_power_change_1",
    "deviation_from_15min_mean",
    "deviation_from_60min_mean",

    "active_power_change_rate_log",
    "rolling_zscore_15_log",
    "rolling_zscore_60_log"
]