import numpy as np
import lightgbm as lgb
from app.ml.predictor import Predictor


class LgbmPredictor(Predictor):
    def __init__(self, booster: lgb.Booster, version: str):
        self._booster = booster
        self._version = version

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X)

    @property
    def version(self) -> str:
        return self._version

    @property
    def booster(self) -> lgb.Booster:
        return self._booster
