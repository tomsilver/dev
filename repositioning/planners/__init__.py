"""Helper function for creating planners."""

from dynamics.base_model import RepositioningDynamicsModel
from structs import RepositioningSceneConfig

from .base_planner import RepositioningPlanner
from .predictive_sampling_planner import PredictiveSamplingPlanner
from .random_planner import RandomRepositioningPlanner


def create_planner(
    name: str,
    scene_config: RepositioningSceneConfig,
    dynamics: RepositioningDynamicsModel,
    seed: int,
) -> RepositioningPlanner:
    """Helper function for creating planners."""

    if name == "random":
        return RandomRepositioningPlanner(
            scene_config=scene_config,
            dynamics=dynamics,
            seed=seed,
        )

    if name == "predictive-sampling":
        return PredictiveSamplingPlanner(
            scene_config=scene_config, dynamics=dynamics, seed=seed
        )

    raise NotImplementedError
