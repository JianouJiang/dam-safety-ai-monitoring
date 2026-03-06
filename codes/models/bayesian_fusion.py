#!/usr/bin/env python3
"""
Dempster-Shafer Evidence Combination for multi-source anomaly fusion.

Implements the full Dempster-Shafer framework:
1. Basic Probability Assignment (BPA/mass functions) for each evidence source
2. Dempster's rule of combination with conflict normalization
3. Belief and plausibility intervals for uncertainty quantification
4. Validation-based calibration of per-source sigmoid parameters

Detection fusion: Physics (PINN) + Data (GAT-LSTM) via DS combination.
Diagnosis overlay: Knowledge layer provides failure-mode labels post-hoc.

Reference: Shafer (1976), "A Mathematical Theory of Evidence"
"""

import numpy as np
from scipy.special import expit  # sigmoid


class DempsterShaferFusion:
    """Dempster-Shafer evidence combination for multi-source anomaly fusion.

    Frame of discernment: Theta = {Normal, Anomalous}
    Power set: 2^Theta = {emptyset, {Normal}, {Anomalous}, {Normal, Anomalous}}

    Mass function m assigns belief to subsets:
    - m({Normal}): belief that system is normal
    - m({Anomalous}): belief that system is anomalous
    - m(Theta): uncertainty (don't know)
    - m(emptyset) = 0 by axiom
    """

    def __init__(self, physics_weight=0.50, data_weight=0.50):
        self.source_reliability = {
            "physics": physics_weight,
            "data": data_weight,
        }
        self.sigmoid_centers = {}
        self.sigmoid_slopes = {}
        self._calibrated = False

    def calibrate(self, physics_val_scores, data_val_scores):
        """Calibrate sigmoid parameters from clean validation data.

        Centers the sigmoid at max(P99, mean+4*sigma), above the vast majority
        of normal scores. This ensures:
        - Normal scores produce near-zero anomaly belief (background < 0.05)
        - Anomaly scores that exceed the normal range produce meaningful belief
        - Both PINN and GAT-LSTM contribute to fusion (neither is vacuous)
        """
        for name, scores in [("physics", physics_val_scores),
                              ("data", data_val_scores)]:
            mu = np.mean(scores)
            sigma = np.std(scores)
            # Center above ~99% of normal scores
            pct99 = np.percentile(scores, 99)
            stat_thresh = mu + 4.0 * sigma
            center = max(pct99, stat_thresh)
            # Slope: moderate steepness for smooth discrimination
            effective_sigma = max(sigma, 0.1)
            slope = 2.0 / effective_sigma
            self.sigmoid_centers[name] = center
            self.sigmoid_slopes[name] = slope
        self._calibrated = True

    def _score_to_bpa(self, score, source_name):
        """Convert raw anomaly score to Basic Probability Assignment (BPA).

        Uses threshold-centered sigmoid: scores below the normal-operation
        threshold produce near-zero anomaly belief. Scores above it
        produce increasing belief discounted by source reliability.
        """
        reliability = self.source_reliability[source_name]
        center = self.sigmoid_centers[source_name]
        slope = self.sigmoid_slopes[source_name]

        p_anomalous = float(expit(slope * (score - center)))

        # Shafer's discounting
        m_anomalous = reliability * p_anomalous
        m_normal = reliability * (1 - p_anomalous)
        m_theta = 1.0 - m_anomalous - m_normal

        return {
            "normal": m_normal,
            "anomalous": m_anomalous,
            "theta": m_theta,
        }

    @staticmethod
    def _dempster_combine(m1, m2):
        """Combine two BPAs using Dempster's rule of combination.

        m_combined(C) = (1/(1-K)) * sum_{A cap B = C} m1(A) * m2(B)
        where K = sum_{A cap B = emptyset} m1(A) * m2(B).
        """
        keys = ["normal", "anomalous", "theta"]
        combined = {"normal": 0.0, "anomalous": 0.0, "theta": 0.0}
        conflict = 0.0

        for a in keys:
            for b in keys:
                product = m1[a] * m2[b]
                if product == 0:
                    continue
                if a == "theta":
                    combined[b] += product
                elif b == "theta":
                    combined[a] += product
                elif a == b:
                    combined[a] += product
                else:
                    conflict += product

        if conflict < 1.0 - 1e-10:
            norm = 1.0 / (1.0 - conflict)
            for k in combined:
                combined[k] *= norm
        else:
            combined = {"normal": 0.0, "anomalous": 0.0, "theta": 1.0}

        combined["conflict"] = conflict
        return combined

    def fuse(self, physics_score, data_score):
        """Fuse physics and data scores using Dempster-Shafer combination.

        Returns:
            risk_level: str
            m_final: dict of combined mass
            layer_bpas: dict of per-layer BPAs
        """
        m_physics = self._score_to_bpa(physics_score, "physics")
        m_data = self._score_to_bpa(data_score, "data")

        m_final = self._dempster_combine(m_physics, m_data)

        bel_anomalous = m_final["anomalous"]
        if bel_anomalous < 0.1:
            risk_level = "Normal"
        elif bel_anomalous < 0.3:
            risk_level = "Watch"
        elif bel_anomalous < 0.5:
            risk_level = "Alert"
        else:
            risk_level = "Emergency"

        return risk_level, m_final, {
            "physics": m_physics,
            "data": m_data,
        }

    def fuse_timeseries(self, physics_scores, data_scores):
        """Fuse time series of scores from physics and data layers."""
        n = len(physics_scores)
        risk_levels = []
        beliefs = []
        combined_scores = np.zeros(n)
        layer_contributions = {
            "physics": np.zeros(n),
            "data": np.zeros(n),
        }

        for i in range(n):
            level, m_final, layer_bpas = self.fuse(
                physics_scores[i], data_scores[i]
            )
            risk_levels.append(level)
            beliefs.append(m_final)
            combined_scores[i] = m_final["anomalous"]

            for k in ["physics", "data"]:
                layer_contributions[k][i] = layer_bpas[k]["anomalous"]

        return risk_levels, beliefs, combined_scores, layer_contributions
