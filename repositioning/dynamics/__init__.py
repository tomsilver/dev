"""Helper function to create dynamics models."""

import pybullet as p
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

from ..robots.human import HumanArm6DoF
from ..robots.panda import PandaPybulletRobotLimbRepo
from ..structs import RepositioningSceneConfig
from .base_model import RepositioningDynamicsModel
from .math_model import MathRepositioningDynamicsModel
from .pybullet_constraint_model import \
    PybulletConstraintRepositioningDynamicsModel


def create_robot(
    name: str, physics_client_id: int, base_pose: Pose, joint_positions: JointPositions
) -> SingleArmPyBulletRobot:
    """Create a robot from its name."""
    if name == "human-arm-6dof":
        return HumanArm6DoF(
            physics_client_id, base_pose=base_pose, home_joint_positions=joint_positions
        )

    if name == "panda-limb-repo":
        return PandaPybulletRobotLimbRepo(
            physics_client_id, base_pose=base_pose, home_joint_positions=joint_positions
        )

    return create_pybullet_robot(
        name,
        physics_client_id,
        base_pose=base_pose,
        home_joint_positions=joint_positions,
    )


def create_dynamics_model(
    name: str,
    physics_client_id: int,
    scene_config: RepositioningSceneConfig,
) -> RepositioningDynamicsModel:
    """Helper function to create dynamics models."""

    active_arm = create_robot(
        scene_config.active_name, physics_client_id, scene_config.active_base_pose
    )
    passive_arm = create_robot(scene_config.passive_name)
    dt = scene_config.dt

    if name == "math":
        return MathRepositioningDynamicsModel(active_arm, passive_arm, dt)

    if name == "pybullet-constraint":
        return PybulletConstraintRepositioningDynamicsModel(active_arm, passive_arm, dt)

    raise NotImplementedError
