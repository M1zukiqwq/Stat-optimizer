from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _copy_matrix(matrix: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in matrix]


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(matrix)
    if n == 0:
        return []

    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        pivot_value = augmented[pivot][col]
        if abs(pivot_value) < 1e-12:
            raise ValueError("singular matrix encountered while fitting model")

        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        factor = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= factor

        for row in range(n):
            if row == col:
                continue
            row_factor = augmented[row][col]
            if row_factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= row_factor * augmented[col][j]

    return [augmented[i][n] for i in range(n)]


@dataclass
class RidgeMultiOutputRegressor:
    alpha: float = 1.0
    weights: Optional[List[List[float]]] = None
    bias: Optional[List[float]] = None

    def fit(self, features: List[List[float]], targets: List[List[float]]) -> None:
        if not features or not targets:
            raise ValueError("features and targets must be non-empty")
        if len(features) != len(targets):
            raise ValueError("features and targets must have same sample count")

        sample_count = len(features)
        feature_dim = len(features[0])
        output_dim = len(targets[0])

        xtx = [[0.0] * (feature_dim + 1) for _ in range(feature_dim + 1)]
        xty = [[0.0] * output_dim for _ in range(feature_dim + 1)]

        for sample_idx in range(sample_count):
            x = features[sample_idx]
            y = targets[sample_idx]
            if len(x) != feature_dim:
                raise ValueError("inconsistent feature dimension")
            if len(y) != output_dim:
                raise ValueError("inconsistent target dimension")

            x_with_bias = x + [1.0]

            for i in range(feature_dim + 1):
                xi = x_with_bias[i]
                for j in range(feature_dim + 1):
                    xtx[i][j] += xi * x_with_bias[j]
                for out_idx in range(output_dim):
                    xty[i][out_idx] += xi * y[out_idx]

        for idx in range(feature_dim):
            xtx[idx][idx] += self.alpha

        coefficient_matrix = [[0.0] * output_dim for _ in range(feature_dim + 1)]

        for out_idx in range(output_dim):
            rhs = [xty[row_idx][out_idx] for row_idx in range(feature_dim + 1)]
            solution = _solve_linear_system(_copy_matrix(xtx), rhs)
            for row_idx, coefficient in enumerate(solution):
                coefficient_matrix[row_idx][out_idx] = coefficient

        self.weights = [
            [coefficient_matrix[row_idx][out_idx] for row_idx in range(feature_dim)]
            for out_idx in range(output_dim)
        ]
        self.bias = [coefficient_matrix[feature_dim][out_idx] for out_idx in range(output_dim)]

    def predict(self, features: List[List[float]]) -> List[List[float]]:
        if self.weights is None or self.bias is None:
            raise ValueError("model is not fitted")

        output_dim = len(self.weights)
        predictions: List[List[float]] = []

        for x in features:
            row_prediction = []
            for out_idx in range(output_dim):
                value = self.bias[out_idx]
                weight_row = self.weights[out_idx]
                if len(weight_row) != len(x):
                    raise ValueError("feature dimension mismatch in predict")
                for w, xi in zip(weight_row, x):
                    value += w * xi
                row_prediction.append(value)
            predictions.append(row_prediction)
        return predictions

    def save(self, path: str, metadata: Optional[Dict[str, object]] = None) -> None:
        if self.weights is None or self.bias is None:
            raise ValueError("model is not fitted")

        payload = {
            "alpha": self.alpha,
            "weights": self.weights,
            "bias": self.bias,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "RidgeMultiOutputRegressor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(alpha=float(payload.get("alpha", 1.0)))
        model.weights = payload["weights"]
        model.bias = payload["bias"]
        return model

    @staticmethod
    def load_metadata(path: str) -> Dict[str, object]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return {}
