"""Examples:

Multirun:
    python main2.py -m seed=1,2,3 model=fizz,buzz

Single run with override:
    python main2.py seed=4 model=fizz model.fizz=100
"""

import hydra
import abc
import numpy as np
import logging
from omegaconf import DictConfig


class BaseModel(abc.ABC):
    """Base class for models."""

    @abc.abstractmethod
    def predict(self, x: float) -> float:
        """Predict an output from an input."""


class FizzModel(BaseModel):
    """A fizz model takes in a fizz parameter."""

    def __init__(self, fizz: float) -> None:
        self._fizz = fizz

    def predict(self, x: float) -> float:
        return x * self._fizz


class BuzzModel(BaseModel):
    """A buzz model takes in a buzz parameter."""

    def __init__(self, buzz: float) -> None:
        self._buzz = buzz

    def predict(self, x: float) -> float:
        return self._buzz
    


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(f"Running with {cfg.seed} and {cfg.model}")
    model = hydra.utils.instantiate(cfg.model)
    # TODO this fails, maybe due to double import issue? Hopefully gets
    # cleared up when this is a really package.
    # assert isinstance(model, BaseModel)
    rng = np.random.default_rng(cfg.seed)
    print(model.predict(rng.uniform()))


if __name__ == "__main__":
    _main()
