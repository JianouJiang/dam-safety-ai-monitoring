#!/usr/bin/env python3
"""
Baseline anomaly detection methods for comparison.

1. Threshold-based: industry standard — flag when sensor exceeds ±N sigma
2. HST (Hydrostatic-Seasonal-Time): standard dam monitoring statistical model
3. Isolation Forest: unsupervised anomaly detection
4. LSTM Autoencoder: deep learning reconstruction-based method
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest


class ThresholdDetector:
    """Industry-standard threshold method: flag readings beyond N sigma."""

    def __init__(self, n_sigma=3.0):
        self.n_sigma = n_sigma
        self.means = None
        self.stds = None

    def fit(self, sensor_data):
        self.means = sensor_data.mean(axis=0)
        self.stds = sensor_data.std(axis=0) + 1e-8

    def get_anomaly_scores(self, sensor_data):
        z_scores = np.abs((sensor_data - self.means) / self.stds)
        return np.max(z_scores, axis=1)

    def predict(self, sensor_data):
        scores = self.get_anomaly_scores(sensor_data)
        return (scores > self.n_sigma).astype(int)


class HSTModel:
    """Hydrostatic-Seasonal-Time statistical model.

    δ = Σ aᵢhⁱ + b₁sin(2πt) + b₂cos(2πt) + b₃sin(4πt) + b₄cos(4πt)
        + c₁θ + c₂θ² + c₃ln(1+θ)

    where h = normalized water level, t = time in years, θ = time since start.
    """

    def __init__(self, n_sigma=3.0):
        self.n_sigma = n_sigma
        self.coefficients = None
        self.residual_stats = None

    def _build_features(self, water_level, n_days):
        h = water_level / 185.0  # normalized by dam height
        t = n_days / 365.25  # years

        features = np.column_stack([
            h, h ** 2, h ** 3, h ** 4,
            np.sin(2 * np.pi * t), np.cos(2 * np.pi * t),
            np.sin(4 * np.pi * t), np.cos(4 * np.pi * t),
            t, t ** 2, np.log(1 + t),
            np.ones(len(h)),
        ])
        return features

    def fit(self, water_level, n_days, sensor_data):
        X = self._build_features(water_level, n_days)
        # Least squares fit for each sensor
        self.coefficients = np.linalg.lstsq(X, sensor_data, rcond=None)[0]
        residuals = sensor_data - X @ self.coefficients
        self.residual_stats = {
            "mean": residuals.mean(axis=0),
            "std": residuals.std(axis=0) + 1e-8,
        }

    def get_anomaly_scores(self, water_level, n_days, sensor_data):
        X = self._build_features(water_level, n_days)
        predictions = X @ self.coefficients
        residuals = sensor_data - predictions
        z_scores = np.abs((residuals - self.residual_stats["mean"]) / self.residual_stats["std"])
        return np.max(z_scores, axis=1)

    def predict(self, water_level, n_days, sensor_data):
        scores = self.get_anomaly_scores(water_level, n_days, sensor_data)
        return (scores > self.n_sigma).astype(int)


class IsolationForestDetector:
    """Isolation Forest for multivariate anomaly detection."""

    def __init__(self, contamination=0.05, n_estimators=200, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=8,
        )

    def fit(self, sensor_data):
        self.model.fit(sensor_data)

    def get_anomaly_scores(self, sensor_data):
        # score_samples returns negative scores (more negative = more anomalous)
        return -self.model.score_samples(sensor_data)

    def predict(self, sensor_data):
        return (self.model.predict(sensor_data) == -1).astype(int)


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time series anomaly detection."""

    def __init__(self, input_dim=30, hidden_dim=64, n_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        enc_out, (h, c) = self.encoder(x)
        # Decode: use encoder outputs reversed + hidden state
        decoder_input = torch.zeros(x.size(0), x.size(1), self.decoder.input_size,
                                     device=x.device)
        dec_out, _ = self.decoder(decoder_input, (h, c))
        recon = self.output_layer(dec_out)
        return recon


class LSTMAutoencoderDetector:
    """Training wrapper for LSTM Autoencoder anomaly detection."""

    def __init__(self, n_sensors=30, seq_len=30, device="cpu",
                 lr=1e-3, epochs=80, batch_size=64):
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.train_stats = None

    def _create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i:i + self.seq_len])
        return np.array(sequences)

    def fit(self, sensor_data):
        self.scaler_mean = sensor_data.mean(axis=0)
        self.scaler_std = sensor_data.std(axis=0) + 1e-8
        data_norm = (sensor_data - self.scaler_mean) / self.scaler_std

        sequences = self._create_sequences(data_norm)
        X_t = torch.FloatTensor(sequences).to(self.device)

        self.model = LSTMAutoencoder(
            input_dim=self.n_sensors, hidden_dim=64, n_layers=2
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            perm = np.random.permutation(len(X_t))
            epoch_loss = 0
            n_batches = 0

            for start in range(0, len(X_t), self.batch_size):
                end = min(start + self.batch_size, len(X_t))
                idx = perm[start:end]
                optimizer.zero_grad()
                recon = self.model(X_t[idx])
                loss = nn.MSELoss()(recon, X_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                print(f"  [LSTM-AE] Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.6f}")

        self.model.load_state_dict(best_state)

        # Compute training residual stats
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_t).cpu().numpy()
        residuals = np.abs(sequences - recon)
        self.train_stats = {
            "mean": residuals.mean(axis=(0, 1)),
            "std": residuals.std(axis=(0, 1)) + 1e-8,
        }

    def get_anomaly_scores(self, sensor_data):
        data_norm = (sensor_data - self.scaler_mean) / self.scaler_std
        sequences = self._create_sequences(data_norm)
        X_t = torch.FloatTensor(sequences).to(self.device)

        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                end = min(start + self.batch_size, len(X_t))
                recon = self.model(X_t[start:end]).cpu().numpy()
                residuals = np.abs(sequences[start:end] - recon)
                # Score based on last timestep
                last_res = residuals[:, -1, :]
                z_scores = (last_res - self.train_stats["mean"]) / self.train_stats["std"]
                scores = np.max(z_scores, axis=1)
                all_scores.append(scores)

        scores = np.concatenate(all_scores)
        # Pad
        pad = np.zeros(self.seq_len - 1)
        return np.concatenate([pad, scores])
