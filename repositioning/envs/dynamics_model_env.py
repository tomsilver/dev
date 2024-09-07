"""An environment implemented with a dynamics model."""

import abc

from pybullet_helpers.camera import capture_image
from pybullet_helpers.gui import create_gui_connection

from dynamics import create_dynamics_model
from dynamics.base_model import RepositioningDynamicsModel
from structs import (Image, JointTorques, RepositioningSceneConfig,
                     RepositioningState)

from .repositioning_env import RepositioningEnv


class DynamicsModelEnv(RepositioningEnv):
    """An environment implemented with a dynamics model."""

    def __init__(
        self,
        dynamics_name: str,
        scene_config: RepositioningSceneConfig | None = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if scene_config is None:
            scene_config = self._get_default_scene_config()
        self._scene_config = scene_config

        self._physics_client_id = create_gui_connection(
            camera_target=self._scene_config.camera_target,
            camera_distance=self._scene_config.camera_distance,
            camera_pitch=self._scene_config.camera_pitch,
        )
        self._dynamics_model = create_dynamics_model(
            dynamics_name, self._physics_client_id, self._scene_config
        )

    @abc.abstractmethod
    def _get_default_scene_config(self) -> RepositioningSceneConfig:
        """Subclasses should define a default scene config."""

    def get_scene_config(self) -> RepositioningSceneConfig:
        return self._scene_config

    def get_state(self) -> RepositioningState:
        return self._dynamics_model.get_state()

    def reset(self, state: RepositioningState) -> None:
        self._dynamics_model.reset(state)

    def step(self, action: JointTorques) -> None:
        self._dynamics_model.step(action)

    @property
    def _initial_state(self) -> RepositioningState:
        return RepositioningState(
            self._scene_config.active_init_joint_positions,
            self._scene_config.active_init_joint_velocities,
            self._scene_config.passive_init_joint_positions,
            self._scene_config.passive_init_joint_velocities,
        )

    def get_goal(self) -> RepositioningState:
        return RepositioningState(
            self._scene_config.active_goal_joint_positions,
            self._scene_config.active_goal_joint_velocities,
            self._scene_config.passive_goal_joint_positions,
            self._scene_config.passive_goal_joint_velocities,
        )

    def get_torque_limits(self) -> tuple[JointTorques, JointTorques]:
        return (
            self._scene_config.torque_lower_limits,
            self._scene_config.torque_upper_limits,
        )

    def render(self) -> Image:
        return capture_image(
            self._physics_client_id,
            camera_distance=self._scene_config.camera_distance,
            camera_pitch=self._scene_config.camera_pitch,
            camera_target=self._scene_config.camera_target,
            image_height=self._scene_config.image_height,
            image_width=self._scene_config.image_width,
        )
