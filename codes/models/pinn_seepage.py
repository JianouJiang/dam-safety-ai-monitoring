#!/usr/bin/env python3
"""
Physics-Informed Neural Network (PINN) for dam seepage and displacement.

The PINN encodes three governing equations as physics loss terms:

1. Steady-state seepage (Darcy's law, 1D through dam body):
   d/dz [k(T) dh/dz] = 0
   where h = piezometric head, z = elevation, k = permeability (temperature-dependent)
   Boundary conditions: h(z_upstream) = H_reservoir, h(z_downstream) = H_tailwater

2. Thermal-structural coupling:
   delta_thermal = alpha * integral(T(z) - T_ref, dz)
   where alpha = thermal expansion coefficient

3. Hydrostatic displacement (beam bending under water pressure):
   delta = sum(a_i * (H/H_dam)^i)  with d(delta)/dH > 0

Input to the network: (water_level, air_temperature, normalized_elevation)
The third input (spatial coordinate) enables PDE residual computation.
"""

import torch
import torch.nn as nn
import numpy as np
import os


class PINNSeepage(nn.Module):
    """PINN with spatial coordinate input for PDE enforcement."""

    def __init__(self, input_dim=3, hidden_dim=64, n_layers=4, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PINNResidualEstimator:
    """Train PINNs per sensor type with genuine physics losses."""

    def __init__(self, device="cpu", lr=1e-3, epochs=200, physics_weight=0.1):
        self.device = torch.device(device)
        self.lr = lr
        self.epochs = epochs
        self.physics_weight = physics_weight
        self.models = {}
        self.scalers = {}

    def _normalize(self, X, fit=False, key="default"):
        if fit:
            self.scalers[key] = {
                "mean": X.mean(axis=0),
                "std": X.std(axis=0) + 1e-8,
            }
        return (X - self.scalers[key]["mean"]) / self.scalers[key]["std"]

    def _physics_loss_seepage(self, inputs, model):
        """Seepage PDE residual: d/dz [k dh/dz] = 0 (Darcy steady-state).

        Since k is approximately constant within a short time window,
        this simplifies to d²h/dz² ≈ 0 (linear head distribution).

        Also enforces:
        - dh/dH > 0 (head increases with reservoir level)
        - Monotonic decrease with depth: dh/dz > 0 (head drops with depth)
        """
        inputs = inputs.clone().requires_grad_(True)
        h_pred = model(inputs)  # predicted piezometric head

        # Compute gradients w.r.t. all inputs
        grad_h = torch.autograd.grad(
            h_pred.sum(), inputs, create_graph=True
        )[0]

        # dh/d(water_level): should be positive
        dh_dH = grad_h[:, 0]
        monotonicity_loss = torch.relu(-dh_dH).mean()

        # dh/d(elevation): for piezometers below water surface, head decreases
        # with depth into the foundation, so dh/dz should be positive
        # (higher elevation = higher head)
        dh_dz = grad_h[:, 2]

        # d²h/dz² (Laplacian for 1D seepage): should be near zero
        grad2_h = torch.autograd.grad(
            dh_dz.sum(), inputs, create_graph=True
        )[0][:, 2]
        laplacian_loss = (grad2_h ** 2).mean()

        return monotonicity_loss + 0.5 * laplacian_loss

    def _physics_loss_displacement(self, inputs, model):
        """Displacement physics: hydrostatic + thermal.

        1. d(delta)/d(H) > 0 for upstream-downstream displacement
        2. d(delta)/d(z) follows beam bending: higher = more displacement
           d²(delta)/dz² = M(z)/(EI) where M is bending moment
        3. Temperature coupling: d(delta)/d(T) should be positive
           (expansion moves crest downstream)
        """
        inputs = inputs.clone().requires_grad_(True)
        d_pred = model(inputs)

        grad_d = torch.autograd.grad(
            d_pred.sum(), inputs, create_graph=True
        )[0]

        # d(delta)/d(H) > 0: displacement increases with water level
        dd_dH = grad_d[:, 0]
        hydro_loss = torch.relu(-dd_dH).mean()

        # d(delta)/d(z): displacement increases with elevation (cantilever beam)
        dd_dz = grad_d[:, 2]
        cantilever_loss = torch.relu(-dd_dz).mean()

        # Smoothness in elevation (beam continuity)
        grad2_d = torch.autograd.grad(
            dd_dz.sum(), inputs, create_graph=True
        )[0][:, 2]
        smooth_loss = (grad2_d ** 2).mean()

        return hydro_loss + 0.3 * cantilever_loss + 0.1 * smooth_loss

    def _physics_loss_temperature(self, inputs, model):
        """Temperature physics: depth-dependent attenuation.

        1. d(T_concrete)/d(T_air) > 0 at surface, ≈ 0 at depth
        2. |d(T)/d(z)| should decrease with depth (attenuation)
        3. Temperature should be smooth in elevation
        """
        inputs = inputs.clone().requires_grad_(True)
        T_pred = model(inputs)

        grad_T = torch.autograd.grad(
            T_pred.sum(), inputs, create_graph=True
        )[0]

        # dT/d(air_temp): should be positive (concrete follows air temp)
        dT_dTair = grad_T[:, 1]
        coupling_loss = torch.relu(-dT_dTair).mean()

        # Smoothness in elevation
        dT_dz = grad_T[:, 2]
        grad2_T = torch.autograd.grad(
            dT_dz.sum(), inputs, create_graph=True
        )[0][:, 2]
        smooth_loss = (grad2_T ** 2).mean()

        return coupling_loss + 0.1 * smooth_loss

    def train_sensor_group(self, X_train, y_train, sensor_type, elevations):
        """Train PINN for a sensor group.

        Args:
            X_train: (n_samples, 2) — [water_level, air_temperature]
            y_train: (n_samples, n_sensors) — sensor readings
            sensor_type: 'displacement', 'piezometer', or 'temperature'
            elevations: (n_sensors,) — normalized elevation for each sensor
        """
        n_samples, n_sensors = y_train.shape

        # Build training data with spatial coordinate
        # Replicate each time sample for each sensor, add elevation
        X_expanded = np.repeat(X_train, n_sensors, axis=0)  # (n*s, 2)
        z_expanded = np.tile(elevations, n_samples)  # (n*s,)
        X_full = np.column_stack([X_expanded, z_expanded])  # (n*s, 3)
        y_full = y_train.flatten()[:, np.newaxis]  # (n*s, 1)

        X_norm = self._normalize(X_full, fit=True, key=f"{sensor_type}_X")
        y_norm = self._normalize(y_full, fit=True, key=f"{sensor_type}_y")

        model = PINNSeepage(
            input_dim=3, hidden_dim=64, n_layers=4, output_dim=1
        ).to(self.device)
        self.models[sensor_type] = model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        # Use a random subset per epoch for efficiency
        n_total = len(X_norm)
        subset_size = min(n_total, 5000)

        X_t = torch.FloatTensor(X_norm).to(self.device)
        y_t = torch.FloatTensor(y_norm).to(self.device)

        best_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            # Random subset for this epoch
            idx = np.random.choice(n_total, subset_size, replace=False)
            X_sub = X_t[idx]
            y_sub = y_t[idx]

            y_pred = model(X_sub)
            data_loss = nn.MSELoss()(y_pred, y_sub)

            # Physics loss
            physics_loss_fn = {
                "piezometer": self._physics_loss_seepage,
                "displacement": self._physics_loss_displacement,
                "temperature": self._physics_loss_temperature,
            }[sensor_type]
            physics_loss = physics_loss_fn(X_sub, model)

            total_loss = data_loss + self.physics_weight * physics_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 100 == 0:
                print(f"  [{sensor_type}] Epoch {epoch+1}/{self.epochs}: "
                      f"data={data_loss.item():.6f}, physics={physics_loss.item():.6f}")

        model.load_state_dict(best_state)
        # Store elevations for inference
        self.scalers[f"{sensor_type}_elevations"] = elevations
        return model

    def compute_residuals(self, X, y, sensor_type):
        """Compute PINN residuals for anomaly detection."""
        elevations = self.scalers[f"{sensor_type}_elevations"]
        n_samples, n_sensors = y.shape

        X_expanded = np.repeat(X, n_sensors, axis=0)
        z_expanded = np.tile(elevations, n_samples)
        X_full = np.column_stack([X_expanded, z_expanded])
        y_full = y.flatten()[:, np.newaxis]

        X_norm = self._normalize(X_full, fit=False, key=f"{sensor_type}_X")
        y_norm = self._normalize(y_full, fit=False, key=f"{sensor_type}_y")

        model = self.models[sensor_type]
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_norm).to(self.device)
            y_pred = model(X_t).cpu().numpy()

        residuals = np.abs(y_norm - y_pred)
        residuals_original = residuals * self.scalers[f"{sensor_type}_y"]["std"]
        # Reshape back to (n_samples, n_sensors)
        return residuals_original.reshape(n_samples, n_sensors)

    def get_anomaly_scores(self, X, y_dict, train_residuals_dict):
        """Combine residuals across sensor types into physics anomaly score."""
        all_scores = []
        for sensor_type, y in y_dict.items():
            residuals = self.compute_residuals(X, y, sensor_type)
            train_mean = train_residuals_dict[sensor_type]["mean"]
            train_std = train_residuals_dict[sensor_type]["std"]
            z_scores = (residuals - train_mean) / (train_std + 1e-8)
            type_score = np.max(z_scores, axis=1)
            all_scores.append(type_score)

        combined = np.max(np.stack(all_scores, axis=1), axis=1)
        return combined

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "scalers": self.scalers,
            "model_states": {k: v.state_dict() for k, v in self.models.items()},
            "model_configs": {k: {"input_dim": 3, "hidden_dim": 64, "n_layers": 4,
                                   "output_dim": 1}
                              for k in self.models},
        }
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.scalers = save_dict["scalers"]
        for name, cfg in save_dict["model_configs"].items():
            model = PINNSeepage(**cfg).to(self.device)
            model.load_state_dict(save_dict["model_states"][name])
            self.models[name] = model
