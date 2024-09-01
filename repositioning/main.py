"""Experimental script for one robot moving another."""

import time
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as iio
import numpy as np
from pybullet_helpers.camera import capture_image
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
    SingleArmPyBulletRobot,
    SingleArmPyBulletRobot,
    Callable[[float], list[float]],
    dict[str, Any],
]:

    if scenario == "two-link":
        camera_kwargs = {"camera_distance": 2.0, "camera_pitch": -60}
        physics_client_id = create_gui_connection(**camera_kwargs)

        pad = 0.2 # add pad to prevent contact forces
        active_arm_base_pose = Pose((-np.sqrt(2) - pad, 0.0, 0.0))
        active_arm_home_joint_positions = [-np.pi / 4, np.pi / 2]
        active_arm = TwoLinkPyBulletRobot(
            physics_client_id,
            base_pose=active_arm_base_pose,
            home_joint_positions=active_arm_home_joint_positions,
        )

        passive_arm_base_pose = Pose((np.sqrt(2) + pad, 0.0, 0.0))
        passive_arm_home_joint_positions = [np.pi / 2 + np.pi / 4, np.pi / 2]
        passive_arm = TwoLinkPyBulletRobot(
            physics_client_id,
            base_pose=passive_arm_base_pose,
            home_joint_positions=passive_arm_home_joint_positions,
        )

        def _torque_fn(t: float) -> list[float]:
            if t < 0.5:
                return [0.1, -0.1]
            if t < 1.0:
                return [-0.1, 0.1]
            if t < 1.5:
                return [0.2, 0.0]
            if t < 2.0:
                return [-0.2, 0.0]
            if t < 2.5:
                return [0.0, 0.2]
            if t < 3.0:
                return [0.0, -0.2]
            return [0.0, 0.0]

        return active_arm, passive_arm, _torque_fn, camera_kwargs

    if scenario == "panda-human":
        robot_init_pos = (0.8, -0.1, 0.5)
        human_init_pos = (0.15, 0.1, 1.4)

        camera_kwargs = {
            "camera_target": human_init_pos,
            "camera_distance": 1.75,
            "camera_pitch": -50,
        }
        physics_client_id = create_gui_connection(**camera_kwargs)
        camera_kwargs["image_height"] = 512
        camera_kwargs["image_width"] = 512

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

        return robot, human, _torque_fn, camera_kwargs

    raise NotImplementedError


def _main(scenario: str, dynamics: str, make_video: bool, video_dt: float) -> None:
    dt = 1e-3
    T = 5.0
    t = 0.0

    active_arm, passive_arm, torque_fn, camera_kwargs = _create_scenario(scenario)
    dynamics_model = _create_dynamics_model(dynamics, active_arm, passive_arm, dt)

    if make_video:
        imgs = [capture_image(active_arm.physics_client_id, **camera_kwargs)]

    while t < T:
        torque = torque_fn(t)
        dynamics_model.step(torque)
        time.sleep(dt)
        t += dt
        if make_video and ((t + dt) // video_dt) > (t // video_dt):
            imgs.append(capture_image(active_arm.physics_client_id, **camera_kwargs))

    if make_video:
        video_outfile = Path(__file__).parent / f"{scenario}_{dynamics}.mp4"
        iio.mimsave(video_outfile, imgs, fps=int(1.0 / video_dt))
        print(f"Wrote out to {video_outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="two-link")
    parser.add_argument("--dynamics", type=str, default="math")
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--video_dt", type=float, default=1e-1)
    args = parser.parse_args()

    _main(args.scenario, args.dynamics, args.make_video, args.video_dt)