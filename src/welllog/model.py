"""HistGradientBoosting lithology model and well-grouped validation split."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit


def group_train_test_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "WELL",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positional train/test indices with disjoint groups."""
    if group_column not in frame:
        raise KeyError(group_column)
    if frame[group_column].nunique(dropna=False) < 2:
        raise ValueError("At least two wells are required for a grouped split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(splitter.split(frame, groups=frame[group_column].astype(str)))
    return train, test


@dataclass
class LithologyModel:
    """Small deterministic wrapper around HistGradientBoostingClassifier."""

    random_state: int = 42
    max_iter: int = 150
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    class_weight: str | dict | None = "balanced"
    model_: HistGradientBoostingClassifier | None = field(default=None, init=False, repr=False)
    feature_names_: tuple[str, ...] = field(default=(), init=False)

    def fit(self, features: pd.DataFrame, target: pd.Series | np.ndarray) -> LithologyModel:
        """Fit the classifier after validating a numeric feature table."""
        matrix = self._validate_features(features, fitting=True)
        labels = np.asarray(target)
        if matrix.shape[0] != labels.shape[0]:
            raise ValueError("Feature and target row counts differ")
        if pd.isna(labels).any():
            raise ValueError("Target contains missing values")
        self.model_ = HistGradientBoostingClassifier(
            random_state=self.random_state,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            class_weight=self.class_weight,
            early_stopping=False,
        )
        self.model_.fit(matrix, labels)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._require_model().predict(self._validate_features(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self._require_model().predict_proba(self._validate_features(features))

    @property
    def classes_(self) -> np.ndarray:
        return self._require_model().classes_

    def uncertainty(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return confidence, uncertainty, normalized entropy, and prediction."""
        probabilities = self.predict_proba(features)
        confidence = probabilities.max(axis=1)
        entropy = -(probabilities * np.log(np.clip(probabilities, 1e-15, 1))).sum(axis=1)
        if probabilities.shape[1] > 1:
            entropy /= np.log(probabilities.shape[1])
        return pd.DataFrame(
            {
                "prediction": self.classes_[probabilities.argmax(axis=1)],
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "entropy": entropy,
            },
            index=features.index,
        )

    def _validate_features(self, features: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if fitting:
            if features.columns.duplicated().any():
                raise ValueError("Feature names must be unique")
            self.feature_names_ = tuple(features.columns)
        elif not self.feature_names_:
            raise RuntimeError("Model has not been fitted")
        elif tuple(features.columns) != self.feature_names_:
            raise ValueError("Feature columns/order do not match fitted model")
        non_numeric = [
            column for column in features if not pd.api.types.is_numeric_dtype(features[column])
        ]
        if non_numeric:
            raise TypeError(f"Non-numeric features: {non_numeric}")
        return features.replace([np.inf, -np.inf], np.nan).astype(float)

    def _require_model(self) -> HistGradientBoostingClassifier:
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted")
        return self.model_
