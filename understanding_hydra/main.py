"""Run python main.py -m seed=1,2,3 model=foo,bar
"""

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
import abc
import numpy as np
import logging


class BaseModel(abc.ABC):
    """Base class for models."""

    @abc.abstractmethod
    def predict(self, x: float) -> float:
        """Predict an output from an input."""

@dataclass
class FooConfig:
    """Config for FooModel."""

    foo: float


class FooModel(BaseModel):
    """A foo model takes in a foo parameter."""

    def __init__(self, foo: float) -> None:
        self._foo = foo

    def predict(self, x: float) -> float:
        return x * self._foo
    

@dataclass
class BarConfig:
    """Config for BarModel."""

    bar: float


class BarModel(BaseModel):
    """A bar model takes in a bar parameter."""

    def __init__(self, bar: float) -> None:
        self._bar = bar

    def predict(self, x: float) -> float:
        return x + self._bar


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def _main(cfg: DictConfig) -> None:
    logging.info(f"Running with {cfg.seed} and {cfg.model}")
    model = hydra.utils.instantiate(cfg.model)
    rng = np.random.default_rng(cfg.seed)
    inputs = rng.uniform(0, 1, size=100)
    outputs = model.predict(inputs)
    print(outputs)


if __name__ == "__main__":
    _main()
