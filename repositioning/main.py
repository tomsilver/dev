"""Experimental script for one robot moving another."""

import time
from typing import Callable

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.robots.two_link import TwoLinkPyBulletRobot
from scipy.spatial.transform import Rotation

from dynamics.base_model import RepositioningDynamicsModel
from dynamics.math_model import MathRepositioningDynamicsModel
from dynamics.pybullet_constraint_model import \
    PybulletConstraintRepositioningDynamicsModel
from robots.human import HumanArm6DoF
from robots.panda import PandaPybulletRobotLimbRepo


def _create_dynamics_model(
    name: str,
    active_arm: SingleArmPyBulletRobot,
    passive_arm: SingleArmPyBulletRobot,
    dt: float,
) -> RepositioningDynamicsModel:
    if name == "math":
        return MathRepositioningDynamicsModel(active_arm, passive_arm, dt)

    if name == "pybullet-constraint":
        return PybulletConstraintRepositioningDynamicsModel(active_arm, passive_arm, dt)

    raise NotImplementedError


def _create_scenario(
    scenario: str,
) -> tuple[
    SingleArmPyBulletRobot, SingleArmPyBulletRobot, Callable[[float], list[float]]
]:

    if scenario == "two-link":
        physics_client_id = create_gui_connection(camera_distance=2.0, camera_pitch=-40)

        active_arm_base_pose = Pose((-np.sqrt(2), 0.0, 0.0))
        active_arm_home_joint_positions = [-np.pi / 4, np.pi / 2]
        active_arm = TwoLinkPyBulletRobot(
            physics_client_id,
            base_pose=active_arm_base_pose,
            home_joint_positions=active_arm_home_joint_positions,
        )

        passive_arm_base_pose = Pose((np.sqrt(2), 0.0, 0.0))
        passive_arm_home_joint_positions = [np.pi / 2 + np.pi / 4, np.pi / 2]
        passive_arm = TwoLinkPyBulletRobot(
            physics_client_id,
            base_pose=passive_arm_base_pose,
            home_joint_positions=passive_arm_home_joint_positions,
        )

        def _torque_fn(t: float) -> list[float]:
            if t < 0.05:
                return [0.0, 1.0]
            return [0.0] * 2

        return active_arm, passive_arm, _torque_fn

    if scenario == "panda-human":
        robot_init_pos = (0.8, -0.1, 0.5)
        human_init_pos = (0.15, 0.1, 1.4)

        physics_client_id = create_gui_connection(
            camera_target=robot_init_pos, camera_distance=1.75, camera_pitch=-50
        )

        robot_init_orn_obj = Rotation.from_euler("xyz", [0, 0, np.pi])
        robot_base_pose = Pose(robot_init_pos, robot_init_orn_obj.as_quat())
        human_init_orn_obj = Rotation.from_euler("xyz", [np.pi, 0, 0])
        human_base_pose = Pose(human_init_pos, human_init_orn_obj.as_quat())
        robot = PandaPybulletRobotLimbRepo(physics_client_id, base_pose=robot_base_pose)
        human = HumanArm6DoF(physics_client_id, base_pose=human_base_pose)
        robot_init_joints = [
            0.94578431,
            -0.89487842,
            -1.67534487,
            -0.34826698,
            1.73607292,
            0.14233887,
        ]
        human_init_joints = [
            1.43252278,
            -0.81111486,
            -0.42373363,
            0.49931369,
            -1.17420521,
            0.37122887,
        ]
        robot.set_joints(robot_init_joints)
        human.set_joints(human_init_joints)

        def _torque_fn(t: float) -> list[float]:
            if t < 0.5:
                return [0.1, 0.0, 0.0, 0.01, 0.0, 0.0]
            if t < 1:
                return [-0.1, 0.0, 0.0, 0.01, 0.0, 0.0]
            return [0.0] * 6

        return robot, human, _torque_fn

    raise NotImplementedError


def _main(scenario: str, dynamics: str) -> None:
    dt = 1e-3
    T = 10.0
    t = 0.0

    active_arm, passive_arm, torque_fn = _create_scenario(scenario)
    dynamics_model = _create_dynamics_model(dynamics, active_arm, passive_arm, dt)

    while t < T:
        torque = torque_fn(t)
        dynamics_model.step(torque)
        time.sleep(dt)
        t += dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="two-link")
    parser.add_argument("--dynamics", type=str, default="math")
    args = parser.parse_args()

    _main(args.scenario, args.dynamics)
