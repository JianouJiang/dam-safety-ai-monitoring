#!/usr/bin/env python3
"""
Graph-Attention LSTM for multivariate dam sensor anomaly detection.

Architecture:
1. Graph Attention Network (GAT) per Velickovic et al. (2018) with
   learnable attention coefficients and multi-head attention.
2. LSTM processes the temporal sequence of graph-attended features.
3. Output: per-timestep anomaly score based on reconstruction error.

The learned attention weights reveal which sensor pairs contribute
to detected anomalies — interpretable spatial propagation patterns.

Efficiency: Uses additive attention with split src/dst parameters
(avoids full N×N×2d expansion). All timesteps processed in one batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """Single-head GAT layer (Velickovic et al. 2018).

    Uses additive attention: e_ij = LeakyReLU(a_src^T W h_i + a_dst^T W h_j)
    This avoids creating the full (batch, N, N, 2d) tensor.
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)
        self.alpha = alpha

    def forward(self, x, adj):
        """
        Args:
            x: (batch, n_nodes, in_features)
            adj: (n_nodes, n_nodes) binary/weighted adjacency
        Returns:
            out: (batch, n_nodes, out_features)
            attn: (batch, n_nodes, n_nodes) attention coefficients
        """
        h = self.W(x)  # (batch, n, d)
        # Additive attention scores
        e_src = self.a_src(h)  # (batch, n, 1)
        e_dst = self.a_dst(h)  # (batch, n, 1)
        e = e_src + e_dst.transpose(1, 2)  # (batch, n, n) broadcasting
        e = F.leaky_relu(e, self.alpha)

        # Mask non-adjacent pairs
        mask = (adj == 0).unsqueeze(0)  # (1, n, n)
        e = e.masked_fill(mask, float("-inf"))

        attn = F.softmax(e, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)  # isolated nodes

        out = torch.bmm(attn, h)  # (batch, n, d)
        return out, attn


class MultiHeadGAT(nn.Module):
    """Multi-head graph attention: concatenates heads then projects."""

    def __init__(self, in_features, out_features, n_heads=2, alpha=0.2):
        super().__init__()
        assert out_features % n_heads == 0
        self.head_dim = out_features // n_heads
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, self.head_dim, alpha)
            for _ in range(n_heads)
        ])

    def forward(self, x, adj):
        head_outs = []
        attns = []
        for head in self.heads:
            out, attn = head(x, adj)
            head_outs.append(out)
            attns.append(attn)
        out = torch.cat(head_outs, dim=-1)  # (batch, n, out_features)
        attn_avg = torch.stack(attns).mean(dim=0)  # avg attention across heads
        return F.elu(out), attn_avg


class GATLSTM(nn.Module):
    """GAT-LSTM for multivariate time series anomaly detection.

    All timesteps are processed through GAT in a single batched call
    by reshaping (batch, seq, n, f) -> (batch*seq, n, f).
    """

    def __init__(self, n_sensors=30, sensor_features=1, gat_hidden=16,
                 n_heads=2, lstm_hidden=64, lstm_layers=2, dropout=0.1,
                 projection_dim=64):
        super().__init__()
        self.n_sensors = n_sensors
        self.gat = MultiHeadGAT(sensor_features, gat_hidden, n_heads)
        # Project flattened GAT output before LSTM
        self.projection = nn.Linear(n_sensors * gat_hidden, projection_dim)
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(lstm_hidden, n_sensors * sensor_features)
        self.gat_hidden = gat_hidden

    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, n_sensors, sensor_features)
            adj: (n_sensors, n_sensors)
        Returns:
            recon: (batch, seq_len, n_sensors)
            attn_last: (batch, n_sensors, n_sensors) attention at last timestep
        """
        batch, seq_len, n_sensors, feat = x.shape

        # Apply GAT to all timesteps at once
        x_flat = x.reshape(batch * seq_len, n_sensors, feat)
        gat_out, attn_flat = self.gat(x_flat, adj)
        # gat_out: (batch*seq, n, gat_hidden)

        gat_seq = gat_out.reshape(batch, seq_len, -1)  # (batch, seq, n*gat_hidden)
        gat_seq = F.relu(self.projection(gat_seq))  # (batch, seq, proj_dim)

        lstm_out, _ = self.lstm(gat_seq)
        recon = self.output_layer(lstm_out)  # (batch, seq, n_sensors)

        # Attention at last timestep for interpretability
        attn_last = attn_flat.reshape(batch, seq_len, n_sensors, n_sensors)[:, -1]
        return recon, attn_last


class GATLSTMAnomalyDetector:
    """Training and inference wrapper for GAT-LSTM anomaly detection."""

    def __init__(self, n_sensors=30, seq_len=14, device="cpu",
                 lr=1e-3, epochs=30, batch_size=256):
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.train_stats = None
        self.scaler_mean = None
        self.scaler_std = None

    def _create_sequences(self, data):
        n = len(data) - self.seq_len + 1
        idx = np.arange(self.seq_len)[None, :] + np.arange(n)[:, None]
        return data[idx]

    def _add_self_loops(self, adj):
        """Add self-loops to adjacency for GAT (nodes attend to themselves)."""
        return adj + np.eye(adj.shape[0]) * (adj.diagonal() == 0)

    def train(self, sensor_data, adj_matrix):
        self.scaler_mean = sensor_data.mean(axis=0)
        self.scaler_std = sensor_data.std(axis=0) + 1e-8
        data_norm = (sensor_data - self.scaler_mean) / self.scaler_std

        adj_sl = self._add_self_loops(adj_matrix).astype(np.float32)
        adj_t = torch.FloatTensor(adj_sl).to(self.device)

        sequences = self._create_sequences(data_norm)
        n_seq = len(sequences)
        X = sequences[:, :, :, np.newaxis].astype(np.float32)
        y = sequences[:, -1, :].astype(np.float32)

        self.model = GATLSTM(
            n_sensors=self.n_sensors, sensor_features=1,
            gat_hidden=16, n_heads=2, lstm_hidden=64, lstm_layers=2,
            projection_dim=64
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        best_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            perm = np.random.permutation(n_seq)

            for start in range(0, n_seq, self.batch_size):
                end = min(start + self.batch_size, n_seq)
                idx = perm[start:end]
                X_batch = torch.FloatTensor(X[idx]).to(self.device)
                y_batch = torch.FloatTensor(y[idx]).to(self.device)

                optimizer.zero_grad()
                recon, _ = self.model(X_batch, adj_t)
                pred = recon[:, -1, :]
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batches

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  [GAT-LSTM] Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.6f}")

        self.model.load_state_dict(best_state)

        # Compute training residual statistics
        self.model.eval()
        all_train_res = []
        with torch.no_grad():
            for start in range(0, n_seq, self.batch_size):
                end = min(start + self.batch_size, n_seq)
                X_batch = torch.FloatTensor(X[start:end]).to(self.device)
                recon, _ = self.model(X_batch, adj_t)
                all_train_res.append(recon[:, -1, :].cpu().numpy())
        train_recon = np.concatenate(all_train_res)
        train_residuals = np.abs(y - train_recon)
        self.train_stats = {
            "mean": train_residuals.mean(axis=0),
            "std": train_residuals.std(axis=0) + 1e-8,
        }
        self.adj_sl = adj_sl

    def get_anomaly_scores(self, sensor_data, adj_matrix):
        data_norm = (sensor_data - self.scaler_mean) / self.scaler_std
        sequences = self._create_sequences(data_norm)
        X = sequences[:, :, :, np.newaxis].astype(np.float32)
        y = sequences[:, -1, :]
        adj_t = torch.FloatTensor(self.adj_sl).to(self.device)

        self.model.eval()
        all_scores = []
        all_attn = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                X_batch = torch.FloatTensor(X[start:end]).to(self.device)
                recon, attn = self.model(X_batch, adj_t)
                pred = recon[:, -1, :].cpu().numpy()
                attn_np = attn.cpu().numpy()

                residuals = np.abs(y[start:end] - pred)
                z_scores = (residuals - self.train_stats["mean"]) / self.train_stats["std"]
                scores = np.max(z_scores, axis=1)

                all_scores.append(scores)
                all_attn.append(attn_np)

        scores = np.concatenate(all_scores)
        attn_weights = np.concatenate(all_attn)

        # Pad beginning
        pad = np.zeros(self.seq_len - 1)
        scores = np.concatenate([pad, scores])
        pad_attn = np.zeros((self.seq_len - 1, self.n_sensors, self.n_sensors))
        attn_weights = np.concatenate([pad_attn, attn_weights])

        return scores, attn_weights

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "train_stats": self.train_stats,
            "adj_sl": self.adj_sl,
            "n_sensors": self.n_sensors,
            "seq_len": self.seq_len,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.scaler_mean = data["scaler_mean"]
        self.scaler_std = data["scaler_std"]
        self.train_stats = data["train_stats"]
        self.adj_sl = data["adj_sl"]
        self.n_sensors = data["n_sensors"]
        self.seq_len = data["seq_len"]
        self.model = GATLSTM(n_sensors=self.n_sensors).to(self.device)
        self.model.load_state_dict(data["model_state"])
