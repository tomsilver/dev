"""Run: python main.py -m seed=1,2,3 +model=foo,bar"""

import hydra
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import abc
import numpy as np
import logging
from typing import Any


class BaseModel(abc.ABC):
    """Base class for models."""

    @abc.abstractmethod
    def predict(self, x: float) -> float:
        """Predict an output from an input."""


@dataclass
class FooConfig:
    """Config for FooModel."""

    _target_: str = "main.FooModel"
    foo: float = 0.0


class FooModel(BaseModel):
    """A foo model takes in a foo parameter."""

    def __init__(self, foo: float) -> None:
        self._foo = foo

    def predict(self, x: float) -> float:
        return x * self._foo
    

@dataclass
class BarConfig:
    """Config for BarModel."""

    _target_: str = "main.BarModel"
    bar: float = 1.0


class BarModel(BaseModel):
    """A bar model takes in a bar parameter."""

    def __init__(self, bar: float) -> None:
        self._bar = bar

    def predict(self, x: float) -> float:
        return self._bar
    

@dataclass
class ExperimentConfig:
    """Config for a single experiment."""

    seed: int = 0
    model: Any = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)
cs.store(group="model", name="foo", node=FooConfig)
cs.store(group="model", name="bar", node=BarConfig)


@hydra.main(version_base=None, config_name="config")
def _main(cfg: ExperimentConfig) -> None:
    logging.info(f"Running with {cfg.seed} and {cfg.model}")
    model = hydra.utils.instantiate(cfg.model)
    assert isinstance(model, BaseModel)
    rng = np.random.default_rng(cfg.seed)
    print(model.predict(rng.uniform()))


if __name__ == "__main__":
    _main()
