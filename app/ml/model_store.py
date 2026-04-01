from __future__ import annotations

import os
import shutil

import joblib

from app.ml.lgbm_predictor import LgbmPredictor


class ModelStore:
    """Per-mode local .joblib model management. Phase 3: replace with MLflow Registry."""

    def __init__(self, model_dir: str):
        self._dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _current_path(self, battle_type: str) -> str:
        return os.path.join(self._dir, f"matchup_{battle_type}_current.joblib")

    def _backup_path(self, battle_type: str) -> str:
        return os.path.join(self._dir, f"matchup_{battle_type}_backup.joblib")

    def save_current(self, battle_type: str, predictor: LgbmPredictor) -> None:
        current = self._current_path(battle_type)
        if os.path.exists(current):
            shutil.copy2(current, self._backup_path(battle_type))
        joblib.dump({"booster": predictor.booster, "version": predictor.version}, current)

    def load_current(self, battle_type: str) -> LgbmPredictor | None:
        path = self._current_path(battle_type)
        if not os.path.exists(path):
            return None
        data = joblib.load(path)
        return LgbmPredictor(data["booster"], data["version"])

    def has_current(self, battle_type: str) -> bool:
        return os.path.exists(self._current_path(battle_type))
