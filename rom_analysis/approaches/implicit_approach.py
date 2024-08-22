"""Learn a model that maps (input features, point) to bool."""

import abc
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dataset import Dataset, create_classification_data_from_rom_data

from .base_approach import BaseApproach


class ImplicitApproach(BaseApproach):
    """Learn a model that maps (input features, point) to bool."""

    def __init__(self, cache_dir: Path, num_classification_samples: int = 1000) -> None:
        self._num_classification_samples = num_classification_samples
        self._cache_dir = cache_dir

    @abc.abstractmethod
    def _train_from_classification_data(
        self, inputs: NDArray, outputs: NDArray
    ) -> None:
        """Train on classification data."""

    @abc.abstractmethod
    def _predict_from_classification_inputs(
        self,
        inputs: NDArray,
    ) -> NDArray[np.bool_]:
        """Predict from concatenated inputs."""

    def train(self, dataset: Dataset) -> None:
        # Convert into classification data and aggregate.
        combined_inputs: list[NDArray] = []
        outputs: list[bool] = []
        for (subject_id, condition_name), (input_feats, output_arr) in dataset.items():
            data_id = f"{subject_id}_{condition_name}"
            points, labels = create_classification_data_from_rom_data(
                output_arr,
                data_id,
                self._cache_dir,
                num_samples=self._num_classification_samples,
            )
            for point, label in zip(points, labels, strict=True):
                combined_input = np.hstack([input_feats, point])
                combined_inputs.append(combined_input)
                outputs.append(label)
        return self._train_from_classification_data(
            np.array(combined_inputs), np.array(outputs)
        )

    def predict(self, input_features: NDArray, points: NDArray) -> NDArray[np.bool_]:
        combined_inputs: list[NDArray] = []
        for point in points:
            combined_input = np.hstack([input_features, point])
            combined_inputs.append(combined_input)
        return self._predict_from_classification_inputs(np.array(combined_inputs))
