"""Base class for repositioning planner."""

import abc

from ..dynamics.base_model import RepositioningDynamicsModel
from ..structs import JointTorques, RepositioningState


class RepositioningPlanner(abc.ABC):
    """Base class for repositioning planner."""

    def __init__(self, dynamics: RepositioningDynamicsModel) -> None:
        self._dynamics = dynamics
        self._active_arm = dynamics.active_arm
        self._passive_arm = dynamics.passive_arm
        self._dt = dynamics.dt
        self._last_plan: list[JointTorques] | None = None  # for warm starting

    def run(
        self,
        initial_state: RepositioningState,
        goal_state: RepositioningState,
        warm_start: bool = True,
    ) -> list[JointTorques]:
        """Return an open-loop plan.

        Note that the planner can be used with MPC by just calling every
        timestep with warm_start=True.
        """
        self._last_plan = self._run(initial_state, goal_state, warm_start=warm_start)
        return list(self._last_plan)

    @abc.abstractmethod
    def _run(
        self,
        initial_state: RepositioningState,
        goal_state: RepositioningState,
        warm_start: bool = True,
    ) -> list[JointTorques]:
        """Implements the planner logic."""
