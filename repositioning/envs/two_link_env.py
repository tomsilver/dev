"""Environment where a two-link arm is repositioning another."""

from .dynamics_model_env import DynamicsModelEnv
from ..structs import RepositioningSceneConfig

from pybullet_helpers.geometry import Pose
import numpy as np


class TwoLinkRepositioningEnv(DynamicsModelEnv):
    """Environment where a two-link arm is repositioning another."""


    def _get_default_scene_config(self) -> RepositioningSceneConfig:

        active_name = "two-link"
        passive_name = "two-link"

        dt = 1 / 240

        pad = 0.01  # add pad to prevent contact forces
        active_base_pose = Pose((-np.sqrt(2) - pad, 0.0, 0.0))
        passive_base_pose = Pose((np.sqrt(2) + pad, 0.0, 0.0))

        active_init_joint_positions = [-np.pi / 4, np.pi / 2]
        active_init_joint_velocities = [0.0, 0.0]
        passive_init_joint_positions = [np.pi / 2 + np.pi / 4, np.pi / 2]
        passive_init_joint_velocities = [0.0, 0.0]

        # TODO actually set these
        active_goal_joint_positions = [-np.pi / 4, np.pi / 2]
        active_goal_joint_velocities = [0.0, 0.0]
        passive_goal_joint_positions = [np.pi / 2 + np.pi / 4, np.pi / 2]
        passive_goal_joint_velocities = [0.0, 0.0]
        
        torque_lower_limits = [-10.0] * 2
        torque_upper_limits = [10.0] * 2

        camera_target = (0.0, 0.0, 0.0)
        camera_distance = 2.0
        camera_pitch = -60

        image_height = 512
        image_width = 512

        return RepositioningSceneConfig(active_name=active_name,
                                        passive_name=passive_name,
                                        dt=dt,
                                        active_base_pose=active_base_pose,
                                        passive_base_pose=passive_base_pose,
                                        active_init_joint_positions=active_init_joint_positions,
                                        active_init_joint_velocities=active_init_joint_velocities,
                                        passive_init_joint_positions=passive_init_joint_positions,
                                        passive_init_joint_velocities=passive_init_joint_velocities,
                                        active_goal_joint_positions=active_goal_joint_positions,
                                        active_goal_joint_velocities=active_goal_joint_velocities,
                                        passive_goal_joint_positions=passive_goal_joint_positions,
                                        passive_goal_joint_velocities=passive_goal_joint_velocities,
                                        torque_lower_limits=torque_lower_limits,
                                        torque_upper_limits=torque_upper_limits,
                                        camera_target=camera_target,
                                        camera_distance=camera_distance,
                                        camera_pitch=camera_pitch,
                                        image_height=image_height,
                                        image_width=image_width)
