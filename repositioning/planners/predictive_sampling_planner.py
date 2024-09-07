"""A planner that uses predictive sampling."""

import numpy as np
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.trajectory import Trajectory

from structs import JointTorques, RepositioningGoal, RepositioningState

from .base_planner import RepositioningPlanner


class PredictiveSamplingPlanner(RepositioningPlanner):
    """A planner that uses predictive sampling."""

    def __init__(
        self,
        num_rollouts: int = 100,
        noise_scale: float = 1.0,
        num_control_points: int = 10,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_rollouts = num_rollouts
        self._noise_scale = noise_scale
        self._num_control_points = num_control_points
        self._current_plan: Trajectory[JointTorques] | None = None
        self._current_goal: JointPositions | None = None

    def reset(
        self,
        initial_state: RepositioningState,
        goal: RepositioningGoal,
    ) -> None:
        self._current_plan = self._get_initialization()
        self._current_goal = goal

    def step(self, state: RepositioningState) -> JointTorques:
        lower = self._scene_config.torque_lower_limits
        upper = self._scene_config.torque_upper_limits
        scale = np.subtract(upper, lower)
        shift = np.array(lower)
        num_dof = len(lower)
        norm_action = self._rng.uniform(
            0,
            1,
            size=num_dof,
        )
        action = norm_action * scale + shift
        return action.tolist()
    
    def _get_initialization(self) -> Trajectory[JointTorques]:
        import ipdb; ipdb.set_trace()
