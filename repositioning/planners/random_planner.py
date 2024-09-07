"""A very naive 'planner' that just outputs random actions."""

import numpy as np

from structs import JointTorques, RepositioningState

from .base_planner import RepositioningPlanner


class RandomRepositioningPlanner(RepositioningPlanner):
    """A very naive 'planner' that just outputs random actions."""

    def __init__(self, random_plan_length: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._random_plan_length = random_plan_length

    def _run(
        self,
        initial_state: RepositioningState,
        goal_state: RepositioningState,
        warm_start: bool = True,
    ) -> list[JointTorques]:
        lower = self._scene_config.torque_lower_limits
        upper = self._scene_config.torque_upper_limits
        scale = np.subtract(upper, lower)
        shift = np.array(lower)
        num_dof = len(lower)
        norm_samples = self._rng.uniform(
            0,
            1,
            size=(
                self._random_plan_length,
                num_dof,
            ),
        )
        plan = [list(a) for a in norm_samples * scale + shift]
        return plan
