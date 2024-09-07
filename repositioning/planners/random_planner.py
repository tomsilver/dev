"""A very naive 'planner' that just outputs random actions."""

from ..structs import JointTorques, RepositioningState
from .base_planner import RepositioningPlanner


class RandomRepositioningPlanner(RepositioningPlanner):
    """A very naive 'planner' that just outputs random actions."""

    def __init__(
        self,
        torque_lower_limits: JointTorques,
        torque_upper_limits: JointTorques,
        random_plan_length: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def _run(
        self,
        initial_state: RepositioningState,
        goal_state: RepositioningState,
        warm_start: bool = True,
    ) -> list[JointTorques]:
        # TODO
        import ipdb

        ipdb.set_trace()
