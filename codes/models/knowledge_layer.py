#!/usr/bin/env python3
"""
Knowledge Layer: Fuzzy rule engine encoding expert failure-mode signatures.

Encodes 5 canonical dam failure precursor signatures:
1. Piping (internal erosion via seepage)
2. Internal erosion (gradual material loss)
3. Differential settlement
4. Thermal cracking
5. Seepage uplift increase

Each rule evaluates sensor patterns against known failure-mode signatures
and produces an anomaly score [0, 1] with a diagnosis label.

This layer catches failure modes that data-driven models trained on normal
operation will never see — expert knowledge about rare-but-catastrophic events.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class KnowledgeRuleResult:
    rule_name: str
    score: float  # 0 to 1
    triggered: bool
    evidence: str


class DamKnowledgeEngine:
    """Fuzzy rule engine for dam failure-mode detection."""

    def __init__(self, sensor_config, window_size=30):
        self.sensor_config = sensor_config
        self.window_size = window_size
        self.thresholds = {
            "piping": 0.3,
            "internal_erosion": 0.3,
            "differential_settlement": 0.3,
            "thermal_cracking": 0.3,
            "uplift_increase": 0.3,
        }

    def _fuzzy_membership(self, x, low, high):
        """Trapezoidal fuzzy membership: 0 below low, 1 above high, linear between."""
        return np.clip((x - low) / (high - low + 1e-8), 0, 1)

    def _get_sensors_by_type(self, sensor_type):
        return [s for s, cfg in self.sensor_config.items() if cfg["type"] == sensor_type]

    def _get_sensors_by_depth(self, depth):
        return [s for s, cfg in self.sensor_config.items() if cfg.get("depth") == depth]

    def rule_piping(self, data, idx):
        """Piping precursor: simultaneous increase in seepage at multiple
        foundation piezometers with correlation to no change in water level.

        Evidence: piezometric head rising while water level stable or dropping.
        """
        if idx < self.window_size:
            return KnowledgeRuleResult("piping", 0.0, False, "insufficient data")

        window = slice(idx - self.window_size, idx)
        piezometers = self._get_sensors_by_type("piezometer")
        foundation_piez = self._get_sensors_by_depth("foundation") + \
                         self._get_sensors_by_depth("deep_foundation")

        if not foundation_piez:
            return KnowledgeRuleResult("piping", 0.0, False, "no foundation piezometers")

        # Check: piezometric head trend
        piez_trends = []
        for s in foundation_piez:
            if s in data.columns:
                values = data[s].iloc[window].values
                if len(values) > 1:
                    trend = np.polyfit(np.arange(len(values)), values, 1)[0]
                    piez_trends.append(trend)

        if not piez_trends:
            return KnowledgeRuleResult("piping", 0.0, False, "no data")

        # Check: water level trend
        wl_values = data["water_level"].iloc[window].values
        wl_trend = np.polyfit(np.arange(len(wl_values)), wl_values, 1)[0]

        # Piping signature: seepage rising while water level stable/dropping
        mean_piez_trend = np.mean(piez_trends)
        n_rising = sum(1 for t in piez_trends if t > 0.05)

        # Fuzzy scores (calibrated: normal P99~0.11, anomaly mean~0.14)
        piez_rising_score = self._fuzzy_membership(mean_piez_trend, 0.12, 0.30)
        wl_stable_score = self._fuzzy_membership(-wl_trend, -0.1, 0.05)
        multi_sensor_score = self._fuzzy_membership(n_rising / len(piez_trends), 0.2, 0.5)

        score = piez_rising_score * wl_stable_score * multi_sensor_score
        triggered = score > self.thresholds["piping"]

        evidence = (f"piez_trend={mean_piez_trend:.4f}, wl_trend={wl_trend:.4f}, "
                    f"n_rising={n_rising}/{len(piez_trends)}")
        return KnowledgeRuleResult("piping", float(score), triggered, evidence)

    def rule_internal_erosion(self, data, idx):
        """Internal erosion: gradual increase in seepage with decreasing
        curtain efficiency.

        Evidence: gallery/curtain piezometers rising faster than foundation ones.
        """
        if idx < self.window_size:
            return KnowledgeRuleResult("internal_erosion", 0.0, False, "insufficient data")

        window = slice(idx - self.window_size, idx)
        gallery_piez = self._get_sensors_by_depth("gallery") + \
                       self._get_sensors_by_depth("curtain")
        foundation_piez = self._get_sensors_by_depth("foundation")

        gallery_trends = []
        for s in gallery_piez:
            if s in data.columns:
                values = data[s].iloc[window].values
                if len(values) > 1:
                    gallery_trends.append(np.polyfit(np.arange(len(values)), values, 1)[0])

        found_trends = []
        for s in foundation_piez:
            if s in data.columns:
                values = data[s].iloc[window].values
                if len(values) > 1:
                    found_trends.append(np.polyfit(np.arange(len(values)), values, 1)[0])

        if not gallery_trends or not found_trends:
            return KnowledgeRuleResult("internal_erosion", 0.0, False, "insufficient sensors")

        gallery_mean = np.mean(gallery_trends)
        found_mean = np.mean(found_trends)
        differential = gallery_mean - found_mean

        # Calibrated: normal P99~0.09, anomaly values vary
        score = self._fuzzy_membership(differential, 0.05, 0.15)
        triggered = score > self.thresholds["internal_erosion"]

        evidence = f"gallery_trend={gallery_mean:.4f}, foundation_trend={found_mean:.4f}"
        return KnowledgeRuleResult("internal_erosion", float(score), triggered, evidence)

    def rule_differential_settlement(self, data, idx):
        """Differential settlement: displacement divergence between adjacent monoliths.

        Evidence: one monolith drifting while neighbors remain stable.
        """
        if idx < self.window_size:
            return KnowledgeRuleResult("differential_settlement", 0.0, False, "insufficient data")

        window = slice(idx - self.window_size, idx)
        disp_sensors = self._get_sensors_by_type("displacement")

        # Group by monolith
        monolith_trends = {}
        for s in disp_sensors:
            if s in data.columns:
                cfg = self.sensor_config[s]
                mono = cfg["monolith"]
                values = data[s].iloc[window].values
                if len(values) > 1:
                    trend = np.polyfit(np.arange(len(values)), values, 1)[0]
                    monolith_trends.setdefault(mono, []).append(trend)

        if len(monolith_trends) < 2:
            return KnowledgeRuleResult("differential_settlement", 0.0, False, "insufficient monoliths")

        mono_means = {m: np.mean(t) for m, t in monolith_trends.items()}
        overall_mean = np.mean(list(mono_means.values()))
        max_dev = max(abs(v - overall_mean) for v in mono_means.values())

        # Calibrated: normal P95~0.006, anomaly P95~0.062
        score = self._fuzzy_membership(max_dev, 0.02, 0.08)
        triggered = score > self.thresholds["differential_settlement"]

        evidence = f"monolith_trends={mono_means}, max_deviation={max_dev:.4f}"
        return KnowledgeRuleResult("differential_settlement", float(score), triggered, evidence)

    def rule_thermal_cracking(self, data, idx):
        """Thermal cracking: sudden temperature jump at a localized sensor
        inconsistent with seasonal pattern and neighboring sensors.

        Evidence: temperature spike at one sensor while neighbors are normal.
        """
        if idx < self.window_size:
            return KnowledgeRuleResult("thermal_cracking", 0.0, False, "insufficient data")

        window = slice(idx - self.window_size, idx)
        temp_sensors = self._get_sensors_by_type("temperature")

        if not temp_sensors:
            return KnowledgeRuleResult("thermal_cracking", 0.0, False, "no temperature sensors")

        # Compute variability of each sensor in the window
        variabilities = {}
        recent_jumps = {}
        for s in temp_sensors:
            if s in data.columns:
                values = data[s].iloc[window].values
                variabilities[s] = np.std(values)
                # Check for recent jump (last 5 days vs previous 25)
                if len(values) >= 10:
                    recent = np.mean(values[-5:])
                    earlier = np.mean(values[:-5])
                    recent_jumps[s] = abs(recent - earlier)

        if not recent_jumps:
            return KnowledgeRuleResult("thermal_cracking", 0.0, False, "insufficient data")

        mean_jump = np.mean(list(recent_jumps.values()))
        max_jump = max(recent_jumps.values())
        max_sensor = max(recent_jumps, key=recent_jumps.get)

        # Cracking: one sensor jumps while others don't
        isolation_ratio = max_jump / (mean_jump + 1e-8)

        # Calibrated: normal P99~3.1, anomaly P95~6.9
        jump_score = self._fuzzy_membership(max_jump, 3.5, 7.0)
        isolation_score = self._fuzzy_membership(isolation_ratio, 1.5, 3.5)

        score = jump_score * isolation_score
        triggered = score > self.thresholds["thermal_cracking"]

        evidence = f"max_jump={max_jump:.2f}C at {max_sensor}, isolation_ratio={isolation_ratio:.2f}"
        return KnowledgeRuleResult("thermal_cracking", float(score), triggered, evidence)

    def rule_uplift_increase(self, data, idx):
        """Uplift pressure increase: coordinated rise in deep foundation
        piezometers indicating increased hydrostatic uplift.

        Evidence: foundation piezometers rising together, faster than explained
        by water level change.
        """
        if idx < self.window_size:
            return KnowledgeRuleResult("uplift_increase", 0.0, False, "insufficient data")

        window = slice(idx - self.window_size, idx)
        deep_piez = self._get_sensors_by_depth("foundation") + \
                    self._get_sensors_by_depth("deep_foundation")

        piez_trends = []
        for s in deep_piez:
            if s in data.columns:
                values = data[s].iloc[window].values
                if len(values) > 1:
                    piez_trends.append(np.polyfit(np.arange(len(values)), values, 1)[0])

        if not piez_trends:
            return KnowledgeRuleResult("uplift_increase", 0.0, False, "no foundation piezometers")

        wl_values = data["water_level"].iloc[window].values
        wl_trend = np.polyfit(np.arange(len(wl_values)), wl_values, 1)[0]

        mean_piez = np.mean(piez_trends)
        n_rising = sum(1 for t in piez_trends if t > 0.05)
        coordination = n_rising / len(piez_trends)

        # Excess seepage beyond water level explanation
        excess = mean_piez - 0.5 * wl_trend  # ~0.5 is nominal seepage/wl ratio

        # Calibrated: normal P99~0.033, anomaly mean~0.10
        rising_score = self._fuzzy_membership(excess, 0.04, 0.20)
        coord_score = self._fuzzy_membership(coordination, 0.2, 0.6)

        score = rising_score * coord_score
        triggered = score > self.thresholds["uplift_increase"]

        evidence = f"excess_trend={excess:.4f}, coordination={coordination:.2f}"
        return KnowledgeRuleResult("uplift_increase", float(score), triggered, evidence)

    def evaluate_all_rules(self, data, idx):
        """Run all knowledge rules at a given timestep.

        Returns:
            results: dict of rule_name -> KnowledgeRuleResult
            max_score: float, maximum score across all rules
            diagnosis: str, name of highest-scoring rule
        """
        rules = [
            self.rule_piping,
            self.rule_internal_erosion,
            self.rule_differential_settlement,
            self.rule_thermal_cracking,
            self.rule_uplift_increase,
        ]

        results = {}
        for rule_fn in rules:
            result = rule_fn(data, idx)
            results[result.rule_name] = result

        max_score = max(r.score for r in results.values())
        diagnosis = max(results.values(), key=lambda r: r.score).rule_name if max_score > 0 else "normal"

        return results, max_score, diagnosis

    def get_anomaly_scores(self, data):
        """Compute knowledge-based anomaly scores for entire time series.

        Returns:
            scores: (n_days,) anomaly scores
            diagnoses: list of str, per-day diagnosis labels
        """
        n = len(data)
        scores = np.zeros(n)
        diagnoses = ["normal"] * n

        for i in range(n):
            results, max_score, diagnosis = self.evaluate_all_rules(data, i)
            scores[i] = max_score
            diagnoses[i] = diagnosis

        return scores, diagnoses
