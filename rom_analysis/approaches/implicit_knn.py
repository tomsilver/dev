"""Learn a nearest-neighbors model that maps (input features, point) to
bool."""

import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .implicit_approach import ImplicitApproach
from .ml_models import KNeighborsClassifier


class ImplicitKNN(ImplicitApproach):
    """Learn a nearest-neighbors model that maps (input features, point) to
    bool."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._knn = KNeighborsClassifier(seed=0)  # TODO factor out

    def get_name(self) -> str:
        return "implicit-knn"

    def _predict_from_classification_inputs(
        self,
        inputs: NDArray,
    ) -> NDArray[np.bool_]:
        return np.array([self._knn.classify(x) for x in inputs])

    def _train_from_classification_data(
        self, inputs: NDArray, outputs: NDArray
    ) -> None:
        self._knn.fit(inputs, outputs)

    def save(self, filepath: Path) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self._knn, f)
        print(f"Saved model to {filepath}")

    def try_load(self, filepath: Path) -> bool:
        if not filepath.exists():
            return False
        with open(filepath, "rb") as f:
            self._knn = pickle.load(f)
        print(f"Loaded model from {filepath}")
        return True
