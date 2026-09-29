"""
Federated Learning Simulation Module
Simulates on-device local training across 5 decentralized client partitions and central FedAvg aggregation.
DISCLAIMER: Explicitly labeled as "Federated Learning Simulation — Prototype".
"""

import numpy as np
import pandas as pd
from src.impact_model import FEATURE_COLUMNS


class FederatedDevice:
    """
    Represents a client device (e.g., smartphone) maintaining private local data.
    """
    def __init__(self, device_id: str, df_local: pd.DataFrame):
        self.device_id = device_id
        self.df_local = df_local
        self.n_samples = len(df_local)
        self.X = df_local[FEATURE_COLUMNS].values
        self.y = df_local["observed_action"].values
        self.last_local_loss = 0.0
        self.last_weight_delta = None

    def local_train(self, global_weights: np.ndarray, global_bias: float, lr: float = 0.08, local_steps: int = 15):
        """
        Runs local gradient descent on the device's private partition without transmitting raw data.
        Returns:
            delta_w: Weight update vector
            delta_b: Bias update scalar
            local_loss: Binary cross entropy on local shard
        """
        w = global_weights.copy()
        b = float(global_bias)
        n = max(self.n_samples, 1)

        for _ in range(local_steps):
            # Logistic model forward pass
            logits = np.dot(self.X, w) + b
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            
            # Gradients
            err = preds - self.y
            grad_w = np.dot(self.X.T, err) / n + 0.01 * w  # with L2 regularization
            grad_b = np.sum(err) / n
            
            # Local update step
            w -= lr * grad_w
            b -= lr * grad_b

        # Compute post-training loss
        logits = np.dot(self.X, w) + b
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        eps = 1e-7
        loss = -np.mean(self.y * np.log(preds + eps) + (1 - self.y) * np.log(1 - preds + eps))

        delta_w = w - global_weights
        delta_b = b - global_bias

        self.last_local_loss = float(loss)
        self.last_weight_delta = delta_w

        return delta_w, delta_b, float(loss)


class FederatedSimulation:
    """
    Coordinates decentralized client updates and FedAvg aggregation across rounds.
    """
    def __init__(self, df_experiment: pd.DataFrame, n_devices: int = 5, seed: int = 42):
        self.n_devices = n_devices
        self.df_experiment = df_experiment
        self.devices = []
        self._partition_data(seed)
        
        # Initialize global model parameters
        rng = np.random.default_rng(seed)
        self.n_features = len(FEATURE_COLUMNS)
        self.global_weights = rng.normal(0, 0.1, size=self.n_features)
        self.global_bias = 0.0
        self.history = []

    def _partition_data(self, seed: int):
        shuffled = self.df_experiment.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        chunk_size = int(np.ceil(n / self.n_devices))
        device_names = [f"📱 Device {i+1}" for i in range(self.n_devices)]
        for i, name in enumerate(device_names):
            chunk = shuffled.iloc[i * chunk_size : min((i + 1) * chunk_size, n)].reset_index(drop=True)
            self.devices.append(FederatedDevice(name, chunk))

    def run_round(self, round_num: int, lr: float = 0.08, local_steps: int = 15):
        """
        Executes one complete Federated Learning communication round.
        """
        weights_before = self.global_weights.copy()
        bias_before = float(self.global_bias)

        total_samples = sum(dev.n_samples for dev in self.devices)
        delta_w_aggregated = np.zeros_like(self.global_weights)
        delta_b_aggregated = 0.0

        device_results = []

        for dev in self.devices:
            delta_w, delta_b, local_loss = dev.local_train(
                self.global_weights, self.global_bias, lr=lr, local_steps=local_steps
            )
            weight_factor = dev.n_samples / max(total_samples, 1)
            delta_w_aggregated += weight_factor * delta_w
            delta_b_aggregated += weight_factor * delta_b

            local_w = self.global_weights + delta_w
            device_results.append({
                "device_id": dev.device_id,
                "samples": dev.n_samples,
                "local_loss": round(local_loss, 4),
                "delta_norm": round(float(np.linalg.norm(delta_w)), 4),
                "delta_preview": [round(float(x), 4) for x in delta_w],
                "local_weights": [round(float(x), 4) for x in local_w]
            })

        # Apply FedAvg aggregation
        self.global_weights += delta_w_aggregated
        self.global_bias += delta_b_aggregated

        # Evaluate global model on total dataset
        X_all = self.df_experiment[FEATURE_COLUMNS].values
        y_all = self.df_experiment["observed_action"].values
        logits_all = np.dot(X_all, self.global_weights) + self.global_bias
        preds_all = 1.0 / (1.0 + np.exp(-np.clip(logits_all, -20, 20)))
        eps = 1e-7
        global_loss = -np.mean(y_all * np.log(preds_all + eps) + (1 - y_all) * np.log(1 - preds_all + eps))
        global_acc = np.mean((preds_all >= 0.5) == y_all)
        delta_norm = float(np.linalg.norm(delta_w_aggregated))

        round_summary = {
            "round": round_num,
            "global_loss": round(float(global_loss), 4),
            "global_accuracy": round(float(global_acc), 4),
            "delta_norm": round(delta_norm, 4),
            "weights_before": [round(float(x), 4) for x in weights_before],
            "weights_after": [round(float(x), 4) for x in self.global_weights],
            "bias_before": round(bias_before, 4),
            "bias_after": round(float(self.global_bias), 4),
            "device_reports": device_results
        }
        self.history.append(round_summary)
        return round_summary

    def run_multiple_rounds(self, n_rounds: int, lr: float = 0.08, local_steps: int = 15):
        """
        Executes n_rounds of FedAvg sequentially, returning the final round summary.
        """
        res = None
        start_round = len(self.history)
        for i in range(n_rounds):
            res = self.run_round(round_num=start_round + i + 1, lr=lr, local_steps=local_steps)
        return res
