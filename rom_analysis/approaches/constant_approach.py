"""Predict always true or always false."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base_approach import BaseApproach


class ConstantApproach(BaseApproach):
    """Predict always true or always false."""

    def __init__(self, prediction: bool):
        self._prediction = prediction

    def get_name(self) -> str:
        return f"always-{self._prediction}"

    def train(self, dataset: dict[tuple[int, str], tuple[NDArray, NDArray]]) -> None:
        pass

    def predict(self, input_features: NDArray, points: NDArray) -> NDArray[np.bool_]:
        num_predictions = len(points)
        return np.array([self._prediction] * num_predictions)

    def save(self, filepath: Path) -> None:
        pass

    def try_load(self, filepath: Path) -> bool:
        return True  # no loading actually needed
