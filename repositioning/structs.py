"""Common data structures."""

from dataclasses import dataclass
from pybullet_helpers.joint import JointPositions, JointVelocities
from typing import TypeAlias
from pybullet_helpers.geometry import Pose, Pose3D
from numpy.typing import NDArray
import numpy as np


JointTorques: TypeAlias = list[float]
Image: TypeAlias = NDArray[np.uint8]


@dataclass
class RepositioningState:
    """State of a repositioning system: active and passive positions, vels."""

    active_positions: JointPositions
    active_velocities: JointVelocities
    passive_positions: JointPositions
    passive_velocities: JointVelocities


@dataclass
class RepositioningSceneConfig:
    """Static quantities defining the scene, e.g., robot pase positions."""

    active_name: str
    passive_name: str

    dt: float

    active_base_pose: Pose
    passive_base_pose: Pose

    active_init_joint_positions: JointPositions
    active_init_joint_velocities: JointVelocities
    passive_init_joint_positions: JointPositions
    passive_init_joint_velocities: JointVelocities

    active_goal_joint_positions: JointPositions
    active_goal_joint_velocities: JointVelocities
    passive_goal_joint_positions: JointPositions
    passive_goal_joint_velocities: JointVelocities

    torque_lower_limits: JointTorques
    torque_upper_limits: JointTorques

    camera_target: Pose3D
    camera_distance: float
    camera_pitch: float

    image_height: int
    image_width: int

