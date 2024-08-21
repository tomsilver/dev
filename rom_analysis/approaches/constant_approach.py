"""Predict always true or always false."""

import numpy as np
from numpy.typing import NDArray

from .base_approach import BaseApproach


class ConstantApproach(BaseApproach):
    """Predict always true or always false."""

    def __init__(self, prediction: bool):
        self._prediction = prediction

    def train(self, dataset: dict[tuple[int, str], tuple[NDArray, NDArray]]) -> None:
        pass

    def predict(self, input_features: NDArray, points: NDArray) -> NDArray[np.bool_]:
        num_predictions = len(points)
        return np.array([self._prediction] * num_predictions)
