from __future__ import annotations
from typing import Tuple
import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.model import evaluate, predict, train_model


class TestModel:
    @pytest.fixture
    def sample_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42,
        )
        return X, y

    def test_train_model_returns_fitted_estimator(
        self, sample_dataset: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = sample_dataset
        model = train_model(X, y)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_predict_returns_numpy_array(
        self, sample_dataset: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = sample_dataset
        model = train_model(X, y)
        predictions = predict(model, X)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    def test_evaluate_returns_all_expected_metrics(
        self, sample_dataset: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = sample_dataset
        model = train_model(X, y)
        predictions = predict(model, X)
        metrics = evaluate(y, predictions)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0
