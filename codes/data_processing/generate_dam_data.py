#!/usr/bin/env python3
"""
Generate realistic synthetic monitoring data for a concrete gravity dam.

Physics basis:
- Displacement follows the HST model: δ = f(H) + f(S) + f(T) + ε
  where H = hydrostatic (water level), S = seasonal (temperature), T = time (creep)
- Seepage (piezometric head) follows Darcy's law: Q ∝ k·ΔH
  with seasonal temperature effects on permeability
- Temperature follows annual sinusoidal pattern with depth-dependent attenuation

The dam is modeled as a 185m-high concrete gravity dam (Three Gorges scale)
with 30 monitoring sensors arranged in a structured network.

Anomaly injection:
- Seepage increase (piping precursor)
- Differential settlement
- Thermal cracking
- Uplift pressure increase
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta

np.random.seed(42)

# --- Dam geometry and sensor layout ---
DAM_HEIGHT = 185.0  # meters (Three Gorges scale)
DAM_CREST_LENGTH = 2309.0  # meters
NORMAL_WATER_LEVEL = 175.0  # meters (normal pool)
FLOOD_WATER_LEVEL = 183.0  # meters
MIN_WATER_LEVEL = 145.0  # meters (minimum operation)

# Sensor network: 30 sensors across the dam
# Groups: displacement (10), piezometer (10), temperature (10)
NUM_SENSORS = 30

SENSOR_CONFIG = {
    # Displacement sensors (pendulum/GPS) at different elevations and monoliths
    "D01": {"type": "displacement", "elevation": 185.0, "monolith": 1, "direction": "upstream-downstream"},
    "D02": {"type": "displacement", "elevation": 150.0, "monolith": 1, "direction": "upstream-downstream"},
    "D03": {"type": "displacement", "elevation": 120.0, "monolith": 1, "direction": "upstream-downstream"},
    "D04": {"type": "displacement", "elevation": 185.0, "monolith": 2, "direction": "upstream-downstream"},
    "D05": {"type": "displacement", "elevation": 150.0, "monolith": 2, "direction": "upstream-downstream"},
    "D06": {"type": "displacement", "elevation": 120.0, "monolith": 2, "direction": "upstream-downstream"},
    "D07": {"type": "displacement", "elevation": 185.0, "monolith": 3, "direction": "upstream-downstream"},
    "D08": {"type": "displacement", "elevation": 150.0, "monolith": 3, "direction": "upstream-downstream"},
    "D09": {"type": "displacement", "elevation": 185.0, "monolith": 1, "direction": "cross-river"},
    "D10": {"type": "displacement", "elevation": 185.0, "monolith": 3, "direction": "cross-river"},
    # Piezometer sensors (seepage/uplift) at foundation and gallery levels
    "P01": {"type": "piezometer", "elevation": 50.0, "monolith": 1, "depth": "foundation"},
    "P02": {"type": "piezometer", "elevation": 50.0, "monolith": 2, "depth": "foundation"},
    "P03": {"type": "piezometer", "elevation": 50.0, "monolith": 3, "depth": "foundation"},
    "P04": {"type": "piezometer", "elevation": 80.0, "monolith": 1, "depth": "gallery"},
    "P05": {"type": "piezometer", "elevation": 80.0, "monolith": 2, "depth": "gallery"},
    "P06": {"type": "piezometer", "elevation": 80.0, "monolith": 3, "depth": "gallery"},
    "P07": {"type": "piezometer", "elevation": 30.0, "monolith": 1, "depth": "deep_foundation"},
    "P08": {"type": "piezometer", "elevation": 30.0, "monolith": 2, "depth": "deep_foundation"},
    "P09": {"type": "piezometer", "elevation": 30.0, "monolith": 3, "depth": "deep_foundation"},
    "P10": {"type": "piezometer", "elevation": 65.0, "monolith": 2, "depth": "curtain"},
    # Temperature sensors at different elevations and depths
    "T01": {"type": "temperature", "elevation": 180.0, "monolith": 1, "location": "upstream_face"},
    "T02": {"type": "temperature", "elevation": 180.0, "monolith": 2, "location": "upstream_face"},
    "T03": {"type": "temperature", "elevation": 180.0, "monolith": 3, "location": "upstream_face"},
    "T04": {"type": "temperature", "elevation": 130.0, "monolith": 1, "location": "core"},
    "T05": {"type": "temperature", "elevation": 130.0, "monolith": 2, "location": "core"},
    "T06": {"type": "temperature", "elevation": 130.0, "monolith": 3, "location": "core"},
    "T07": {"type": "temperature", "elevation": 80.0, "monolith": 1, "location": "downstream_face"},
    "T08": {"type": "temperature", "elevation": 80.0, "monolith": 2, "location": "downstream_face"},
    "T09": {"type": "temperature", "elevation": 80.0, "monolith": 3, "location": "downstream_face"},
    "T10": {"type": "temperature", "elevation": 50.0, "monolith": 2, "location": "foundation"},
}


def generate_water_level(n_days, start_date):
    """Generate realistic reservoir water level time series.

    Annual cycle: drawdown in spring for flood control, fill in autumn.
    Random fluctuations from rainfall events.
    """
    t = np.arange(n_days)
    # Annual cycle: higher in autumn (Oct), lower in spring (May)
    # Phase shift so minimum is around day 150 (late May)
    annual = NORMAL_WATER_LEVEL + 12.0 * np.sin(2 * np.pi * (t - 120) / 365.25)

    # Multi-year trend: slight overall rise in first 3 years (filling), then stable
    trend = np.minimum(t / 365.25 * 1.5, 4.5)

    # Random rainfall events (short-term fluctuations)
    noise = np.cumsum(np.random.normal(0, 0.15, n_days))
    # Mean-revert the noise
    for i in range(1, n_days):
        noise[i] = 0.97 * noise[i - 1] + np.random.normal(0, 0.15)

    water_level = annual + trend + noise
    # Clip to physical bounds
    water_level = np.clip(water_level, MIN_WATER_LEVEL, FLOOD_WATER_LEVEL)
    return water_level


def generate_air_temperature(n_days, start_date):
    """Generate realistic air temperature for a subtropical dam site (Yichang climate)."""
    t = np.arange(n_days)
    # Annual cycle: ~17°C mean, ~14°C amplitude
    annual = 17.0 + 14.0 * np.sin(2 * np.pi * (t - 100) / 365.25)
    # Daily noise
    noise = np.random.normal(0, 2.5, n_days)
    # Multi-day weather systems
    weather = np.zeros(n_days)
    for i in range(1, n_days):
        weather[i] = 0.85 * weather[i - 1] + np.random.normal(0, 1.5)
    return annual + noise + weather


def hst_displacement(water_level, air_temp, t_days, sensor_cfg, noise_std=0.05):
    """HST model for dam displacement.

    δ = H_component + S_component + T_component + noise

    H (hydrostatic): polynomial in (h/H)^n, n=1..4
    S (seasonal): sin/cos annual + semi-annual
    T (time/creep): logarithmic irreversible component
    """
    h = water_level / DAM_HEIGHT  # normalized water level
    t = t_days / 365.25  # time in years

    elev_factor = sensor_cfg["elevation"] / DAM_HEIGHT
    # Higher sensors = larger displacement
    scale = elev_factor ** 2  # quadratic scaling with height

    # Hydrostatic component: main driver of crest displacement
    h_coeffs = np.array([8.0, -3.5, 2.0, -0.5]) * scale  # mm
    H_comp = sum(h_coeffs[i] * h ** (i + 1) for i in range(4))

    # Seasonal (temperature) component
    S_comp = scale * (
        2.5 * np.sin(2 * np.pi * t) +
        1.2 * np.cos(2 * np.pi * t) +
        0.4 * np.sin(4 * np.pi * t) +
        0.2 * np.cos(4 * np.pi * t)
    )

    # Time/creep component (logarithmic)
    T_comp = scale * 0.8 * np.log(1 + t)

    # Noise
    noise = np.random.normal(0, noise_std * scale, len(t_days))

    return H_comp + S_comp + T_comp + noise


def hst_piezometric_head(water_level, air_temp, t_days, sensor_cfg, noise_std=0.3):
    """Piezometric head model based on simplified seepage physics.

    Seepage pressure at sensor location depends on:
    - Upstream water head (primary driver via Darcy flow)
    - Drainage curtain efficiency (time-dependent degradation)
    - Temperature effects on water viscosity (seasonal)
    """
    h = water_level
    t = t_days / 365.25

    depth_factor = {
        "foundation": 0.45,
        "gallery": 0.30,
        "deep_foundation": 0.55,
        "curtain": 0.20,
    }[sensor_cfg["depth"]]

    # Base piezometric head: fraction of upstream water level
    base = depth_factor * h

    # Seasonal temperature effect on permeability (warmer = more permeable)
    temp_effect = 0.02 * depth_factor * air_temp

    # Time-dependent curtain degradation (slight increase over years)
    degradation = 0.5 * np.log(1 + t) * depth_factor

    # Noise
    noise = np.random.normal(0, noise_std, len(t_days))

    return base + temp_effect + degradation + noise


def generate_concrete_temperature(air_temp, t_days, sensor_cfg, noise_std=0.2):
    """Concrete temperature model with depth-dependent attenuation and phase lag."""
    t = t_days / 365.25

    location_params = {
        "upstream_face": {"attenuation": 0.7, "phase_lag": 15, "mean_offset": -2},
        "core": {"attenuation": 0.15, "phase_lag": 90, "mean_offset": 0},
        "downstream_face": {"attenuation": 0.85, "phase_lag": 10, "mean_offset": 1},
        "foundation": {"attenuation": 0.05, "phase_lag": 180, "mean_offset": -5},
    }
    params = location_params[sensor_cfg["location"]]

    # Long-term mean concrete temperature
    mean_temp = 18.0 + params["mean_offset"]

    # Attenuated and phase-shifted annual cycle
    amplitude = 14.0 * params["attenuation"]
    phase = params["phase_lag"] / 365.25 * 2 * np.pi
    seasonal = amplitude * np.sin(2 * np.pi * t - phase)

    # Slow response to air temperature changes (exponential smoothing)
    alpha = 0.03 * params["attenuation"]
    smoothed_air = np.zeros_like(air_temp)
    smoothed_air[0] = air_temp[0]
    for i in range(1, len(air_temp)):
        smoothed_air[i] = alpha * air_temp[i] + (1 - alpha) * smoothed_air[i - 1]

    concrete_temp = mean_temp + seasonal + 0.1 * params["attenuation"] * (smoothed_air - 17.0)

    # Noise
    noise = np.random.normal(0, noise_std, len(t_days))

    return concrete_temp + noise


def inject_anomalies(data_df, anomaly_config):
    """Inject realistic anomaly signatures into the monitoring data.

    Anomaly types:
    1. Seepage increase (piping precursor): gradual piezometric head rise
    2. Differential settlement: slow displacement drift in one monolith
    3. Thermal cracking: sudden temperature change at crack location
    4. Uplift pressure increase: piezometric head step change
    """
    data = data_df.copy()
    anomaly_labels = np.zeros(len(data), dtype=int)
    anomaly_types = [""] * len(data)

    for anom in anomaly_config:
        start_idx = anom["start_day"]
        duration = anom["duration_days"]
        end_idx = min(start_idx + duration, len(data))

        if anom["type"] == "seepage_increase":
            # Gradual piezometric head increase (piping precursor)
            sensors = [s for s in data.columns if s.startswith("P") and
                       int(s[1:3]) in anom.get("monolith_sensors", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
            for s in sensors:
                t_local = np.arange(end_idx - start_idx)
                magnitude = anom["magnitude"]
                # S-curve onset
                ramp = magnitude * (1 / (1 + np.exp(-0.05 * (t_local - duration * 0.3))))
                data.iloc[start_idx:end_idx, data.columns.get_loc(s)] += ramp[:end_idx - start_idx]

            anomaly_labels[start_idx:end_idx] = 1
            for i in range(start_idx, end_idx):
                anomaly_types[i] = "seepage_increase"

        elif anom["type"] == "differential_settlement":
            # Slow displacement drift in specific monolith
            sensors = [s for s in data.columns if s.startswith("D") and
                       SENSOR_CONFIG.get(s, {}).get("monolith") == anom.get("monolith", 2)]
            for s in sensors:
                t_local = np.arange(end_idx - start_idx)
                magnitude = anom["magnitude"]
                drift = magnitude * (t_local / duration) ** 1.5
                data.iloc[start_idx:end_idx, data.columns.get_loc(s)] += drift[:end_idx - start_idx]

            anomaly_labels[start_idx:end_idx] = 2
            for i in range(start_idx, end_idx):
                anomaly_types[i] = "differential_settlement"

        elif anom["type"] == "thermal_crack":
            # Sudden localized temperature change
            sensors = [s for s in data.columns if s.startswith("T") and
                       SENSOR_CONFIG.get(s, {}).get("monolith") == anom.get("monolith", 1)]
            for s in sensors:
                t_local = np.arange(end_idx - start_idx)
                magnitude = anom["magnitude"]
                # Step with exponential decay
                step = magnitude * np.exp(-t_local / (duration * 0.5))
                data.iloc[start_idx:end_idx, data.columns.get_loc(s)] += step[:end_idx - start_idx]

            anomaly_labels[start_idx:end_idx] = 3
            for i in range(start_idx, end_idx):
                anomaly_types[i] = "thermal_crack"

        elif anom["type"] == "uplift_increase":
            # Step increase in uplift pressure
            sensors = [s for s in data.columns if s.startswith("P") and
                       SENSOR_CONFIG[s]["depth"] in ("foundation", "deep_foundation")]
            for s in sensors:
                t_local = np.arange(end_idx - start_idx)
                magnitude = anom["magnitude"]
                step = magnitude * (1 - np.exp(-t_local / 30))
                data.iloc[start_idx:end_idx, data.columns.get_loc(s)] += step[:end_idx - start_idx]

            anomaly_labels[start_idx:end_idx] = 4
            for i in range(start_idx, end_idx):
                anomaly_types[i] = "uplift_increase"

    data["anomaly_label"] = anomaly_labels
    data["anomaly_type"] = anomaly_types
    return data


def build_adjacency_matrix():
    """Build sensor adjacency matrix based on physical proximity and structural connectivity.

    Edges connect sensors that are:
    1. On the same monolith (structural connectivity)
    2. At similar elevations across adjacent monoliths
    3. Of the same type measuring related quantities
    """
    sensors = list(SENSOR_CONFIG.keys())
    n = len(sensors)
    adj = np.zeros((n, n))

    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i == j:
                continue
            c1, c2 = SENSOR_CONFIG[s1], SENSOR_CONFIG[s2]

            # Same monolith: strong connection
            if c1["monolith"] == c2["monolith"]:
                elev_dist = abs(c1["elevation"] - c2["elevation"]) / DAM_HEIGHT
                adj[i, j] = max(0, 1.0 - elev_dist)

            # Adjacent monoliths: moderate connection
            elif abs(c1["monolith"] - c2["monolith"]) == 1:
                elev_dist = abs(c1["elevation"] - c2["elevation"]) / DAM_HEIGHT
                adj[i, j] = max(0, 0.5 - elev_dist * 0.5)

            # Same sensor type, same elevation band: weak connection
            if c1["type"] == c2["type"]:
                elev_dist = abs(c1["elevation"] - c2["elevation"]) / DAM_HEIGHT
                type_conn = max(0, 0.3 - elev_dist * 0.3)
                adj[i, j] = max(adj[i, j], type_conn)

    # Symmetrize and threshold
    adj = (adj + adj.T) / 2
    adj[adj < 0.1] = 0
    return adj, sensors


def main():
    # --- Configuration ---
    n_years = 10
    start_date = datetime(2014, 1, 1)
    n_days = int(n_years * 365.25)
    t_days = np.arange(n_days)
    dates = [start_date + timedelta(days=int(d)) for d in t_days]

    print(f"Generating {n_years} years of daily monitoring data ({n_days} days)")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"Number of sensors: {NUM_SENSORS}")

    # --- Generate environmental drivers ---
    water_level = generate_water_level(n_days, start_date)
    air_temp = generate_air_temperature(n_days, start_date)

    # --- Generate sensor readings ---
    sensor_data = {}
    sensor_data["date"] = dates
    sensor_data["water_level"] = water_level
    sensor_data["air_temperature"] = air_temp

    for sensor_id, cfg in SENSOR_CONFIG.items():
        if cfg["type"] == "displacement":
            sensor_data[sensor_id] = hst_displacement(
                water_level, air_temp, t_days, cfg)
        elif cfg["type"] == "piezometer":
            sensor_data[sensor_id] = hst_piezometric_head(
                water_level, air_temp, t_days, cfg)
        elif cfg["type"] == "temperature":
            sensor_data[sensor_id] = generate_concrete_temperature(
                air_temp, t_days, cfg)

    data_df = pd.DataFrame(sensor_data)

    # --- Inject anomalies ---
    # Place anomalies in the test period (last 30% of data)
    train_end = int(0.7 * n_days)
    anomaly_config = [
        {
            "type": "seepage_increase",
            "start_day": train_end + 100,
            "duration_days": 30,
            "magnitude": 18.0,  # meters of head (higher for shorter event)
            "monolith_sensors": [1, 2, 4, 5, 7, 8],
        },
        {
            "type": "differential_settlement",
            "start_day": train_end + 250,
            "duration_days": 20,
            "magnitude": 5.0,  # mm
            "monolith": 2,
        },
        {
            "type": "thermal_crack",
            "start_day": train_end + 400,
            "duration_days": 10,
            "magnitude": 10.0,  # degrees C
            "monolith": 1,
        },
        {
            "type": "uplift_increase",
            "start_day": train_end + 550,
            "duration_days": 15,
            "magnitude": 12.0,  # meters of head
        },
    ]

    data_df = inject_anomalies(data_df, anomaly_config)

    # --- Save data ---
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # Main dataset
    csv_path = os.path.join(out_dir, "dam_monitoring_data.csv")
    data_df.to_csv(csv_path, index=False)
    print(f"Saved monitoring data to {csv_path}")
    print(f"  Shape: {data_df.shape}")
    print(f"  Columns: {list(data_df.columns)}")

    # Adjacency matrix
    adj_matrix, sensor_names = build_adjacency_matrix()
    adj_path = os.path.join(out_dir, "adjacency_matrix.npy")
    np.save(adj_path, adj_matrix)
    print(f"Saved adjacency matrix to {adj_path}")
    print(f"  Shape: {adj_matrix.shape}, Non-zero edges: {np.sum(adj_matrix > 0)}")

    # Sensor metadata
    meta_path = os.path.join(out_dir, "sensor_config.json")
    with open(meta_path, "w") as f:
        json.dump(SENSOR_CONFIG, f, indent=2)
    print(f"Saved sensor config to {meta_path}")

    # Anomaly config
    anom_path = os.path.join(out_dir, "anomaly_config.json")
    with open(anom_path, "w") as f:
        json.dump(anomaly_config, f, indent=2)
    print(f"Saved anomaly config to {anom_path}")

    # Summary statistics
    print("\n--- Data Summary ---")
    sensor_cols = [c for c in data_df.columns if c not in
                   ["date", "water_level", "air_temperature", "anomaly_label", "anomaly_type"]]
    print(f"Sensor columns: {sensor_cols}")
    print(f"Training period: {dates[0].date()} to {dates[train_end].date()} ({train_end} days)")
    print(f"Test period: {dates[train_end].date()} to {dates[-1].date()} ({n_days - train_end} days)")
    anomaly_days = (data_df["anomaly_label"] > 0).sum()
    print(f"Anomaly days in test set: {anomaly_days} ({anomaly_days / (n_days - train_end) * 100:.1f}%)")
    for anom_type in ["seepage_increase", "differential_settlement", "thermal_crack", "uplift_increase"]:
        count = (data_df["anomaly_type"] == anom_type).sum()
        print(f"  {anom_type}: {count} days")


if __name__ == "__main__":
    main()
