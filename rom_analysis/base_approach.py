"""Defines the API for an approach."""

import abc

import numpy as np
from numpy.typing import NDArray


class BaseApproach(abc.ABC):
    """An approach."""

    @abc.abstractmethod
    def train(self, dataset: dict[tuple[int, str], tuple[NDArray, NDArray]]) -> None:
        """The dataset maps (subject ID, condition name) to a tuple:

        - A 1D array of "input" features describing the subject / condition
        - A multi-D array of "output" range of motion samples.
        """

    @abc.abstractmethod
    def predict(self, input_features: NDArray, points: NDArray) -> NDArray[np.bool_]:
        """Given input features for a new subject / condition, predict whether
        the given points are within their range of motion."""
