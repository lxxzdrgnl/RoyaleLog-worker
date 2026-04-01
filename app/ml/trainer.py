from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from app.ml.feature_builder import FeatureBuilder
from app.ml.lgbm_predictor import LgbmPredictor
from app.ml.model_store import ModelStore


@dataclass
class TrainResult:
    saved: bool
    model_version: str | None
    train_rows: int
    val_rows: int
    val_accuracy: float
    val_logloss: float
    prev_accuracy: float | None = None
    prev_logloss: float | None = None


class Trainer:
    def __init__(
        self,
        builder: FeatureBuilder,
        store: ModelStore,
        battle_type: str,
        min_child_samples: int = 50,
        early_stopping_rounds: int = 50,
        accuracy_margin: float = 0.005,
    ):
        self._builder = builder
        self._store = store
        self._battle_type = battle_type
        self._min_child_samples = min_child_samples
        self._early_stopping_rounds = early_stopping_rounds
        self._accuracy_margin = accuracy_margin

    def train_from_arrays(self, X: np.ndarray, y: np.ndarray, val_start_index: int) -> TrainResult:
        X_train, y_train = X[:val_start_index], y[:val_start_index]
        X_val, y_val = X[val_start_index:], y[val_start_index:]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error"],
            "verbose": -1,
            "min_child_samples": self._min_child_samples,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_pre_filter": False,
        }

        booster = lgb.train(
            params, train_set, num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(self._early_stopping_rounds, verbose=False)],
        )

        preds = booster.predict(X_val)
        val_accuracy = accuracy_score(y_val, (preds > 0.5).astype(int))
        val_logloss = log_loss(y_val, preds)

        version = f"lgbm-{self._battle_type}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        prev_accuracy, prev_logloss = None, None
        saved = False

        if not self._store.has_current(self._battle_type):
            self._store.save_current(self._battle_type, LgbmPredictor(booster, version))
            saved = True
        else:
            old = self._store.load_current(self._battle_type)
            old_preds = old.predict(X_val)
            prev_accuracy = accuracy_score(y_val, (old_preds > 0.5).astype(int))
            prev_logloss = log_loss(y_val, old_preds)

            if val_accuracy >= prev_accuracy + self._accuracy_margin and val_logloss < prev_logloss:
                self._store.save_current(self._battle_type, LgbmPredictor(booster, version))
                saved = True

        return TrainResult(
            saved=saved,
            model_version=version if saved else None,
            train_rows=len(y_train),
            val_rows=len(y_val),
            val_accuracy=val_accuracy,
            val_logloss=val_logloss,
            prev_accuracy=prev_accuracy,
            prev_logloss=prev_logloss,
        )
