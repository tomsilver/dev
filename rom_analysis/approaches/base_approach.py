"""Defines the API for an approach."""

import abc
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dataset import Dataset


class BaseApproach(abc.ABC):
    """An approach."""

    @abc.abstractmethod
    def get_name(self) -> str:
        """The name of the approach."""

    @abc.abstractmethod
    def train(self, dataset: Dataset) -> None:
        """Train the approach."""

    @abc.abstractmethod
    def predict(self, input_features: NDArray, points: NDArray) -> NDArray[np.bool_]:
        """Given input features for a new subject / condition, predict whether
        the given points are within their range of motion."""

    @abc.abstractmethod
    def save(self, filepath: Path) -> None:
        """Save the trained model."""

    @abc.abstractmethod
    def try_load(self, filepath: Path) -> bool:
        """Attempt to load a saved model."""
