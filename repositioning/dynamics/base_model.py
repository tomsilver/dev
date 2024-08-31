"""Base class for repositioning dynamics model."""

import abc

from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


class RepositioningDynamicsModel(abc.ABC):
    """A model of forward dynamics."""

    def __init__(
        self,
        active_arm: SingleArmPyBulletRobot,
        passive_arm: SingleArmPyBulletRobot,
        dt: float,
    ) -> None:
        self._active_arm = active_arm
        self._passive_arm = passive_arm
        self._dt = dt

    @abc.abstractmethod
    def step(self, torque: list[float]) -> None:
        """Apply torque to the active arm and update in-place."""
