from abc import ABC, abstractmethod
import numpy as np


class Predictor(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...
