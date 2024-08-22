"""Learn a neural network that maps (input features, point) to bool."""

import numpy as np
from numpy.typing import NDArray

from .implicit_approach import ImplicitApproach
from .ml_models import MLPBinaryClassifier


class ImplicitMLP(ImplicitApproach):
    """Learn a neural network that maps (input features, point) to bool."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mlp = MLPBinaryClassifier(
            seed=0,  # TODO pull out
            balance_data=True,
            max_train_iters=1000,
            learning_rate=1e-3,
            n_iter_no_change=1000,
            hid_sizes=[64, 64],
            n_reinitialize_tries=1,
            weight_init="default",
            train_print_every=100,
        )

    def get_name(self) -> str:
        return "implicit-mlp"

    def _predict_from_classification_inputs(
        self,
        inputs: NDArray,
    ) -> NDArray[np.bool_]:
        return np.array([self._mlp.classify(x) for x in inputs])

    def _train_from_classification_data(
        self, inputs: NDArray, outputs: NDArray
    ) -> None:
        self._mlp.fit(inputs, outputs)
