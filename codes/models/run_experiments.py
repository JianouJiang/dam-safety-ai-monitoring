#!/usr/bin/env python3
"""
Main experiment runner: trains all models and evaluates on test data.

Statistical methodology (addressing Statistician Review 001):
1. Three-way split: train (50%) / validation (20%) / test (30%)
2. Thresholds selected on validation set (no test-label leakage)
3. Multi-seed runs (5 seeds) for stochastic methods → mean ± std
4. McNemar's test for pairwise method comparisons
5. Event-level and day-level evaluation metrics
6. All random seeds set for reproducibility
"""

import sys
import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from scipy.stats import chi2 as chi2_dist

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.pinn_seepage import PINNResidualEstimator
from models.gat_lstm import GATLSTMAnomalyDetector
from models.knowledge_layer import DamKnowledgeEngine
from models.bayesian_fusion import DempsterShaferFusion
from models.baselines import (
    ThresholdDetector, HSTModel, IsolationForestDetector,
    LSTMAutoencoderDetector
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = RESULTS_DIR
DAM_HEIGHT = 185.0
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2024]


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    print("=" * 60)
    print("Loading data...")
    data = pd.read_csv(os.path.join(DATA_DIR, "dam_monitoring_data.csv"))
    adj = np.load(os.path.join(DATA_DIR, "adjacency_matrix.npy"))
    with open(os.path.join(DATA_DIR, "sensor_config.json")) as f:
        sensor_config = json.load(f)
    with open(os.path.join(DATA_DIR, "anomaly_config.json")) as f:
        anomaly_config = json.load(f)

    disp_cols = sorted([c for c in data.columns if c.startswith("D")])
    piez_cols = sorted([c for c in data.columns if c.startswith("P")])
    temp_cols = sorted([c for c in data.columns if c.startswith("T")])
    sensor_cols = disp_cols + piez_cols + temp_cols

    print(f"  Data shape: {data.shape}")
    print(f"  Sensors: {len(disp_cols)} disp, {len(piez_cols)} piez, {len(temp_cols)} temp")
    return data, adj, sensor_config, anomaly_config, sensor_cols, disp_cols, piez_cols, temp_cols


def split_data(data, train_frac=0.5, val_frac=0.2):
    """Three-way split: train / validation / test."""
    n = len(data)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    print(f"  Train: {len(train)} days, Val: {len(val)} days, Test: {len(test)} days")
    test_anom = (test["anomaly_label"] > 0).sum()
    print(f"  Test anomaly rate: {test_anom}/{len(test)} = {test_anom/len(test)*100:.1f}%")
    return train, val, test


def get_sensor_elevations(sensor_config, sensor_cols):
    elevations = []
    for s in sensor_cols:
        elev = sensor_config[s]["elevation"] / DAM_HEIGHT
        elevations.append(elev)
    return np.array(elevations)


def select_threshold(val_scores, percentile=99, min_threshold=None):
    """Select threshold from clean validation data.

    Uses max(P99, mean+3*std) to balance false alarm rate against detection.
    This is consistent with the sigmoid calibration center used in the DS
    fusion, ensuring that the binary threshold and the belief calibration
    operate at comparable sensitivity levels.
    """
    pct_thresh = np.percentile(val_scores, 99)
    stat_thresh = np.mean(val_scores) + 3.0 * np.std(val_scores)
    thresh = float(max(pct_thresh, stat_thresh))
    if min_threshold is not None:
        thresh = max(thresh, min_threshold)
    return thresh


def evaluate_at_threshold(y_true, scores, threshold, method_name):
    """Evaluate anomaly detection at a fixed, pre-determined threshold."""
    y_binary = (y_true > 0).astype(int)
    y_pred = (scores > threshold).astype(int)

    f1 = f1_score(y_binary, y_pred, zero_division=0)
    prec = precision_score(y_binary, y_pred, zero_division=0)
    rec = recall_score(y_binary, y_pred, zero_division=0)
    cm = confusion_matrix(y_binary, y_pred, labels=[0, 1])
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    tn = cm[0, 0]
    far = fp / (fp + tn + 1e-10)

    # ROC curve (threshold-independent)
    fpr, tpr, _ = roc_curve(y_binary, scores)
    roc_auc = auc(fpr, tpr)

    # Precision-recall curve (threshold-independent)
    prec_curve, rec_curve, _ = precision_recall_curve(y_binary, scores)
    pr_auc = auc(rec_curve, prec_curve)

    # Event-level detection and lead times
    event_results = compute_event_metrics(y_binary, y_pred, y_true)

    results = {
        "method": method_name,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "false_alarm_rate": far,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "event_detection_rate": event_results["detection_rate"],
        "mean_lead_time": event_results["mean_lead_time"],
        "n_events_detected": event_results["n_detected"],
        "n_events_total": event_results["n_total"],
        "event_details": event_results["details"],
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "prec_curve": prec_curve.tolist(),
        "rec_curve": rec_curve.tolist(),
        "y_pred": y_pred,
    }

    print(f"  {method_name}: F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, "
          f"FAR={far:.3f}, AUC={roc_auc:.3f}, Events={event_results['n_detected']}/{event_results['n_total']}")
    return results


def compute_event_metrics(y_binary, y_pred, y_labels):
    """Event-level detection: per-event detection rate and lead times.

    An event is detected if at least one day within it is flagged.
    Lead time = days from event start to first detection.
    Undetected events are excluded from lead time (reported as miss).
    """
    events = []
    in_event = False
    event_start = 0
    event_type = 0

    for i in range(len(y_binary)):
        if y_binary[i] == 1 and not in_event:
            in_event = True
            event_start = i
            event_type = int(y_labels[i])
        elif y_binary[i] == 0 and in_event:
            in_event = False
            events.append((event_start, i, event_type))
    if in_event:
        events.append((event_start, len(y_binary), event_type))

    details = []
    lead_times = []
    n_detected = 0

    for start, end, etype in events:
        event_preds = y_pred[start:end]
        detected_idx = np.where(event_preds == 1)[0]
        detected = len(detected_idx) > 0
        lead = int(detected_idx[0]) if detected else None
        if detected:
            n_detected += 1
            lead_times.append(lead)
        details.append({
            "start": int(start),
            "end": int(end),
            "type": int(etype),
            "duration": int(end - start),
            "detected": detected,
            "lead_time": lead,
        })

    n_total = len(events)
    return {
        "n_total": n_total,
        "n_detected": n_detected,
        "detection_rate": n_detected / n_total if n_total > 0 else 0.0,
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else float("nan"),
        "details": details,
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test for paired binary classifiers.

    Tests whether two classifiers have the same error rate.
    Returns chi2 statistic and p-value.
    """
    y_binary = (y_true > 0).astype(int)
    correct_a = (y_pred_a == y_binary)
    correct_b = (y_pred_b == y_binary)

    # Discordant pairs
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    if b + c == 0:
        return 0.0, 1.0

    # McNemar with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return float(chi2), float(p_value)


def train_pinn(train, sensor_cols, disp_cols, piez_cols, temp_cols,
               sensor_config, col_map, elevation_map):
    """Train PINN and return model + training residual stats."""
    pinn = PINNResidualEstimator(epochs=200, physics_weight=0.1)
    train_water = train["water_level"].values
    train_temp = train["air_temperature"].values
    X_train_env = np.column_stack([train_water, train_temp])

    for stype in ["displacement", "piezometer", "temperature"]:
        cols = col_map[stype]
        elevs = elevation_map[stype]
        print(f"    Training PINN for {stype} ({len(cols)} sensors)...")
        pinn.train_sensor_group(X_train_env, train[cols].values, stype, elevs)

    train_residuals_dict = {}
    for stype, cols in col_map.items():
        residuals = pinn.compute_residuals(X_train_env, train[cols].values, stype)
        train_residuals_dict[stype] = {
            "mean": residuals.mean(axis=0),
            "std": residuals.std(axis=0) + 1e-8,
        }
    return pinn, train_residuals_dict


def get_pinn_scores(pinn, data, col_map, train_residuals_dict):
    """Compute PINN scores for a dataset."""
    X_env = np.column_stack([data["water_level"].values, data["air_temperature"].values])
    y_dict = {k: data[v].values for k, v in col_map.items()}
    return pinn.get_anomaly_scores(X_env, y_dict, train_residuals_dict)


def train_gat_lstm(train_sensors, adj, n_sensors):
    """Train GAT-LSTM and return model."""
    gat_lstm = GATLSTMAnomalyDetector(
        n_sensors=n_sensors, seq_len=14, epochs=30, batch_size=256)
    gat_lstm.train(train_sensors, adj)
    return gat_lstm


def train_lstm_ae(train_sensors, n_sensors):
    """Train LSTM-AE with its own hyperparameters (not forced to match GAT-LSTM)."""
    lstm_ae = LSTMAutoencoderDetector(
        n_sensors=n_sensors, seq_len=30, epochs=50, batch_size=64)
    lstm_ae.fit(train_sensors)
    return lstm_ae


def run_single_seed(seed, train, val, test, adj, sensor_config,
                    sensor_cols, disp_cols, piez_cols, temp_cols,
                    col_map, elevation_map, threshold_percentile=99):
    """Run one complete experiment with a given seed.

    Returns dict of {method_name: results_dict} for this seed.
    """
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")
    set_all_seeds(seed)

    train_sensors = train[sensor_cols].values
    val_sensors = val[sensor_cols].values
    test_sensors = test[sensor_cols].values
    test_labels = test["anomaly_label"].values

    train_len = len(train)
    val_len = len(val)
    train_days = np.arange(train_len)
    val_days = np.arange(train_len, train_len + val_len)
    test_days = np.arange(train_len + val_len, train_len + val_len + len(test))

    all_results = {}

    # ===== 1. PINN =====
    print(f"  [seed={seed}] Training PINN...")
    pinn, train_res_dict = train_pinn(
        train, sensor_cols, disp_cols, piez_cols, temp_cols,
        sensor_config, col_map, elevation_map)

    pinn_val_scores = get_pinn_scores(pinn, val, col_map, train_res_dict)
    pinn_test_scores = get_pinn_scores(pinn, test, col_map, train_res_dict)
    pinn_thresh = select_threshold(pinn_val_scores, threshold_percentile)
    all_results["PINN (Physics)"] = evaluate_at_threshold(
        test_labels, pinn_test_scores, pinn_thresh, "PINN (Physics)")

    # ===== 2. GAT-LSTM =====
    print(f"  [seed={seed}] Training GAT-LSTM...")
    gat_lstm = train_gat_lstm(train_sensors, adj, len(sensor_cols))

    gat_val_scores, _ = gat_lstm.get_anomaly_scores(val_sensors, adj)
    gat_test_scores, gat_attention = gat_lstm.get_anomaly_scores(test_sensors, adj)
    gat_thresh = select_threshold(gat_val_scores, threshold_percentile)
    all_results["GAT-LSTM (Data)"] = evaluate_at_threshold(
        test_labels, gat_test_scores, gat_thresh, "GAT-LSTM (Data)")

    # ===== 3. Knowledge Layer (deterministic) =====
    print(f"  [seed={seed}] Running Knowledge Layer...")
    knowledge = DamKnowledgeEngine(sensor_config, window_size=30)
    know_val_scores, _ = knowledge.get_anomaly_scores(val)
    know_test_scores, know_diagnoses = knowledge.get_anomaly_scores(test)
    know_thresh = select_threshold(know_val_scores, threshold_percentile)
    all_results["Knowledge Layer"] = evaluate_at_threshold(
        test_labels, know_test_scores, know_thresh, "Knowledge Layer")

    # ===== 4. Dempster-Shafer Fusion (Physics + Data) =====
    # The knowledge layer provides diagnostic labels but does NOT participate
    # in the DS detection fusion. Its scores are anti-informative for detection
    # (validation analysis shows near-zero discrimination), but it provides
    # failure-mode classification for detected anomalies.
    print(f"  [seed={seed}] Running DS Fusion...")
    fusion = DempsterShaferFusion(physics_weight=0.50, data_weight=0.50)
    fusion.calibrate(pinn_val_scores, gat_val_scores)
    # Diagnostic: show calibration parameters and score ranges
    for src in ["physics", "data"]:
        c = fusion.sigmoid_centers[src]
        s = fusion.sigmoid_slopes[src]
        print(f"    {src}: sigmoid center={c:.3f}, slope={s:.3f}")
    print(f"    PINN test scores: mean={np.mean(pinn_test_scores):.3f}, "
          f"max={np.max(pinn_test_scores):.3f}, "
          f"P99={np.percentile(pinn_test_scores, 99):.3f}")
    print(f"    GAT  test scores: mean={np.mean(gat_test_scores):.3f}, "
          f"max={np.max(gat_test_scores):.3f}, "
          f"P99={np.percentile(gat_test_scores, 99):.3f}")

    risk_levels, beliefs, fused_test_scores, layer_contribs = fusion.fuse_timeseries(
        pinn_test_scores, gat_test_scores)
    _, _, fused_val_scores, _ = fusion.fuse_timeseries(
        pinn_val_scores, gat_val_scores)
    # Theoretical bound: max belief from a single-source alarm = w/(1+w).
    # Require fused threshold above this to suppress one-source false alarms
    # and demand corroborating evidence from both detection layers.
    max_w = max(fusion.source_reliability.values())
    single_source_bound = max_w / (1 + max_w)
    fused_thresh = max(select_threshold(fused_val_scores, threshold_percentile),
                       single_source_bound + 0.02)
    print(f"    Fused threshold: {fused_thresh:.4f} "
          f"(val max={np.max(fused_val_scores):.4f}, "
          f"val mean={np.mean(fused_val_scores):.6f}, "
          f"single-source bound={single_source_bound:.4f})")
    all_results["Proposed Framework"] = evaluate_at_threshold(
        test_labels, fused_test_scores, fused_thresh, "Proposed Framework")

    # Add knowledge diagnosis info to layer_contribs for visualization
    layer_contribs["knowledge"] = know_test_scores

    # ===== 5-6. Ablation studies (remove one detection layer) =====
    print(f"  [seed={seed}] Ablation studies...")
    # No Physics: data-only detection
    fusion_no_phys = DempsterShaferFusion(physics_weight=0.50, data_weight=0.50)
    fusion_no_phys.calibrate(pinn_val_scores, gat_val_scores)
    _, _, abl_no_phys_test, _ = fusion_no_phys.fuse_timeseries(
        np.zeros_like(pinn_test_scores), gat_test_scores)
    _, _, abl_no_phys_val, _ = fusion_no_phys.fuse_timeseries(
        np.zeros_like(pinn_val_scores), gat_val_scores)
    t = select_threshold(abl_no_phys_val, threshold_percentile)
    all_results["Ablation: No Physics"] = evaluate_at_threshold(
        test_labels, abl_no_phys_test, t, "Ablation: No Physics")

    # No Data: physics-only detection
    _, _, abl_no_data_test, _ = fusion_no_phys.fuse_timeseries(
        pinn_test_scores, np.zeros_like(gat_test_scores))
    _, _, abl_no_data_val, _ = fusion_no_phys.fuse_timeseries(
        pinn_val_scores, np.zeros_like(gat_val_scores))
    t = select_threshold(abl_no_data_val, threshold_percentile)
    all_results["Ablation: No Data"] = evaluate_at_threshold(
        test_labels, abl_no_data_test, t, "Ablation: No Data")

    # ===== 8. LSTM-AE =====
    print(f"  [seed={seed}] Training LSTM-AE...")
    lstm_ae = train_lstm_ae(train_sensors, len(sensor_cols))
    lstm_ae_val_scores = lstm_ae.get_anomaly_scores(val_sensors)
    lstm_ae_test_scores = lstm_ae.get_anomaly_scores(test_sensors)
    lstm_ae_thresh = select_threshold(lstm_ae_val_scores, threshold_percentile)
    all_results["LSTM-AE"] = evaluate_at_threshold(
        test_labels, lstm_ae_test_scores, lstm_ae_thresh, "LSTM-AE")

    # Store raw scores and attention for the best seed (first one)
    extra = {
        "pinn_test_scores": pinn_test_scores,
        "gat_test_scores": gat_test_scores,
        "know_test_scores": know_test_scores,
        "fused_test_scores": fused_test_scores,
        "gat_attention": gat_attention,
        "risk_levels": risk_levels,
        "know_diagnoses": know_diagnoses,
        "layer_contribs": layer_contribs,
        "pinn_val_scores": pinn_val_scores,
        "gat_val_scores": gat_val_scores,
        "know_val_scores": know_val_scores,
        "lstm_ae_test_scores": lstm_ae_test_scores,
    }
    return all_results, extra


def run_deterministic_baselines(train, val, test, sensor_cols, threshold_percentile=99):
    """Run deterministic baselines once (no seed dependence)."""
    print("\n" + "=" * 60)
    print("Running deterministic baselines...")

    train_sensors = train[sensor_cols].values
    val_sensors = val[sensor_cols].values
    test_sensors = test[sensor_cols].values
    test_labels = test["anomaly_label"].values

    train_water = train["water_level"].values
    val_water = val["water_level"].values
    test_water = test["water_level"].values

    train_len = len(train)
    val_len = len(val)
    train_days = np.arange(train_len)
    val_days = np.arange(train_len, train_len + val_len)
    test_days = np.arange(train_len + val_len, train_len + val_len + len(test))

    results = {}

    # Threshold detector
    threshold_det = ThresholdDetector(n_sigma=3.0)
    threshold_det.fit(train_sensors)
    thr_val = threshold_det.get_anomaly_scores(val_sensors)
    thr_test = threshold_det.get_anomaly_scores(test_sensors)
    t = select_threshold(thr_val, threshold_percentile)
    results["Threshold"] = evaluate_at_threshold(test_labels, thr_test, t, "Threshold")

    # HST
    hst = HSTModel(n_sigma=3.0)
    hst.fit(train_water, train_days, train_sensors)
    hst_val = hst.get_anomaly_scores(val_water, val_days, val_sensors)
    hst_test = hst.get_anomaly_scores(test_water, test_days, test_sensors)
    t = select_threshold(hst_val, threshold_percentile)
    results["HST"] = evaluate_at_threshold(test_labels, hst_test, t, "HST")

    # Isolation Forest
    iforest = IsolationForestDetector(contamination=0.05)
    iforest.fit(train_sensors)
    if_val = iforest.get_anomaly_scores(val_sensors)
    if_test = iforest.get_anomaly_scores(test_sensors)
    t = select_threshold(if_val, threshold_percentile)
    results["Isolation Forest"] = evaluate_at_threshold(test_labels, if_test, t, "Isolation Forest")

    extra_scores = {
        "threshold_val": thr_val, "threshold_test": thr_test,
        "hst_val": hst_val, "hst_test": hst_test,
        "iforest_val": if_val, "iforest_test": if_test,
    }
    return results, extra_scores


def aggregate_multi_seed(all_seed_results, method_names):
    """Aggregate multi-seed results: compute mean ± std for each metric."""
    aggregated = {}
    metrics_keys = ["f1", "precision", "recall", "false_alarm_rate",
                    "roc_auc", "pr_auc", "event_detection_rate", "mean_lead_time"]

    for method in method_names:
        per_seed_values = {k: [] for k in metrics_keys}
        for seed_results in all_seed_results:
            if method in seed_results:
                for k in metrics_keys:
                    val = seed_results[method].get(k, float("nan"))
                    per_seed_values[k].append(val)

        agg = {}
        for k in metrics_keys:
            vals = [v for v in per_seed_values[k] if not np.isnan(v)]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
            else:
                agg[f"{k}_mean"] = float("nan")
                agg[f"{k}_std"] = float("nan")
        agg["method"] = method
        agg["n_seeds"] = len(per_seed_values["f1"])
        aggregated[method] = agg

    return aggregated


def run_experiments():
    start_time = time.time()

    # ===== Load and split data =====
    data, adj, sensor_config, anomaly_config, sensor_cols, disp_cols, piez_cols, temp_cols = load_data()
    train, val, test = split_data(data)

    col_map = {
        "displacement": disp_cols,
        "piezometer": piez_cols,
        "temperature": temp_cols,
    }
    elevation_map = {
        "displacement": get_sensor_elevations(sensor_config, disp_cols),
        "piezometer": get_sensor_elevations(sensor_config, piez_cols),
        "temperature": get_sensor_elevations(sensor_config, temp_cols),
    }

    # ===== Run deterministic baselines (once) =====
    det_results, det_scores = run_deterministic_baselines(
        train, val, test, sensor_cols)

    # ===== Run stochastic methods (multi-seed) =====
    stochastic_methods = [
        "PINN (Physics)", "GAT-LSTM (Data)", "Knowledge Layer",
        "Proposed Framework",
        "Ablation: No Physics", "Ablation: No Data",
        "LSTM-AE",
    ]

    all_seed_results = []
    best_extra = None

    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*60}")
        print(f"  SEED RUN {i+1}/{N_SEEDS}: seed={seed}")
        print(f"{'#'*60}")

        seed_results, extra = run_single_seed(
            seed, train, val, test, adj, sensor_config,
            sensor_cols, disp_cols, piez_cols, temp_cols,
            col_map, elevation_map)
        all_seed_results.append(seed_results)

        if i == 0:
            best_extra = extra

    # ===== Aggregate multi-seed results =====
    print("\n" + "=" * 60)
    print("Aggregating multi-seed results...")
    agg_stochastic = aggregate_multi_seed(all_seed_results, stochastic_methods)

    # ===== Combine deterministic + stochastic =====
    all_methods = list(det_results.keys()) + stochastic_methods
    metrics_keys = ["f1", "precision", "recall", "false_alarm_rate",
                    "roc_auc", "pr_auc", "event_detection_rate", "mean_lead_time"]

    # Build aggregated table
    final_agg = {}
    for method in det_results:
        r = det_results[method]
        final_agg[method] = {"method": method, "n_seeds": 1}
        for k in metrics_keys:
            final_agg[method][f"{k}_mean"] = r.get(k, float("nan"))
            final_agg[method][f"{k}_std"] = 0.0
    final_agg.update(agg_stochastic)

    # ===== McNemar tests (use seed=0 predictions) =====
    print("\nPairwise McNemar tests...")
    test_labels = test["anomaly_label"].values
    seed0 = all_seed_results[0]
    proposed_pred = seed0["Proposed Framework"]["y_pred"]
    mcnemar_results = {}
    comparison_methods = [
        "PINN (Physics)", "GAT-LSTM (Data)", "Knowledge Layer", "LSTM-AE",
    ]
    for method in comparison_methods:
        if method in seed0:
            chi2, pval = mcnemar_test(test_labels, proposed_pred, seed0[method]["y_pred"])
            mcnemar_results[f"Proposed vs {method}"] = {"chi2": chi2, "p_value": pval}
            print(f"  Proposed vs {method}: chi2={chi2:.2f}, p={pval:.4f}")

    # Also compare with deterministic baselines
    for method in det_results:
        chi2, pval = mcnemar_test(test_labels, proposed_pred, det_results[method]["y_pred"])
        mcnemar_results[f"Proposed vs {method}"] = {"chi2": chi2, "p_value": pval}
        print(f"  Proposed vs {method}: chi2={chi2:.2f}, p={pval:.4f}")

    # ===== Print final results table =====
    print("\n" + "=" * 60)
    print("FINAL RESULTS (mean ± std over seeds)")
    print("=" * 60)

    metrics_table = []
    for method in all_methods:
        if method not in final_agg:
            continue
        a = final_agg[method]
        row = {"Method": method}
        for k in metrics_keys:
            mean_val = a.get(f"{k}_mean", float("nan"))
            std_val = a.get(f"{k}_std", 0.0)
            if std_val > 0:
                row[k] = f"{mean_val:.3f}±{std_val:.3f}"
            else:
                row[k] = f"{mean_val:.3f}"
        metrics_table.append(row)

    metrics_df = pd.DataFrame(metrics_table)
    print(metrics_df.to_string(index=False))

    # ===== Save all results =====
    print("\nSaving results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Metrics table (for LaTeX)
    csv_rows = []
    for method in all_methods:
        if method not in final_agg:
            continue
        a = final_agg[method]
        csv_rows.append({
            "Method": method,
            "F1_mean": a.get("f1_mean", ""),
            "F1_std": a.get("f1_std", ""),
            "Precision_mean": a.get("precision_mean", ""),
            "Precision_std": a.get("precision_std", ""),
            "Recall_mean": a.get("recall_mean", ""),
            "Recall_std": a.get("recall_std", ""),
            "FAR_mean": a.get("false_alarm_rate_mean", ""),
            "FAR_std": a.get("false_alarm_rate_std", ""),
            "ROC-AUC_mean": a.get("roc_auc_mean", ""),
            "ROC-AUC_std": a.get("roc_auc_std", ""),
            "PR-AUC_mean": a.get("pr_auc_mean", ""),
            "PR-AUC_std": a.get("pr_auc_std", ""),
            "EventRate_mean": a.get("event_detection_rate_mean", ""),
            "EventRate_std": a.get("event_detection_rate_std", ""),
            "LeadTime_mean": a.get("mean_lead_time_mean", ""),
            "LeadTime_std": a.get("mean_lead_time_std", ""),
        })
    pd.DataFrame(csv_rows).to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"), index=False)

    # Full results (JSON)
    save_results = {
        "aggregated": final_agg,
        "per_seed": [{k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
                       for k, v in sr.items()} for sr in all_seed_results],
        "deterministic": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
                          for k, v in det_results.items()},
        "mcnemar_tests": mcnemar_results,
        "seeds": SEEDS,
        "threshold_percentile": 99,
    }
    with open(os.path.join(RESULTS_DIR, "full_results.json"), "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    # Save score time series from seed 0 for figure generation (test period only)
    ex = best_extra
    scores_df = pd.DataFrame({
        "date": test["date"].values,
        "anomaly_label": test["anomaly_label"].values,
        "pinn_scores": ex["pinn_test_scores"],
        "gat_scores": ex["gat_test_scores"],
        "knowledge_scores": ex["know_test_scores"],
        "fused_scores": ex["fused_test_scores"],
        "threshold_scores": det_scores["threshold_test"],
        "hst_scores": det_scores["hst_test"],
        "iforest_scores": det_scores["iforest_test"],
        "lstm_ae_scores": ex["lstm_ae_test_scores"],
    })
    scores_df.to_csv(os.path.join(RESULTS_DIR, "score_timeseries.csv"), index=False)

    np.save(os.path.join(RESULTS_DIR, "gat_attention_test.npy"), ex["gat_attention"])

    pd.DataFrame({
        "date": test["date"].values,
        "anomaly_label": test["anomaly_label"].values,
        "knowledge_score": ex["know_test_scores"],
        "diagnosis": ex["know_diagnoses"],
        "risk_level": ex["risk_levels"],
    }).to_csv(os.path.join(RESULTS_DIR, "knowledge_diagnoses.csv"), index=False)

    np.savez(os.path.join(RESULTS_DIR, "layer_contributions.npz"),
             physics=ex["layer_contribs"]["physics"],
             data=ex["layer_contribs"]["data"],
             knowledge=ex["layer_contribs"]["knowledge"],
             fused=ex["fused_test_scores"])

    # Event-level results table
    event_rows = []
    for method in all_methods:
        if method not in final_agg:
            continue
        a = final_agg[method]
        event_rows.append({
            "Method": method,
            "EventDetectionRate": a.get("event_detection_rate_mean", ""),
            "MeanLeadTime": a.get("mean_lead_time_mean", ""),
        })
    pd.DataFrame(event_rows).to_csv(
        os.path.join(RESULTS_DIR, "event_metrics.csv"), index=False)

    elapsed = time.time() - start_time
    print(f"\nTotal experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("All results saved to:", RESULTS_DIR)

    return final_agg


if __name__ == "__main__":
    run_experiments()
