"""Experimental script for one robot moving another."""

import time
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as iio
import numpy as np
import pybullet as p
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
from dynamics.snopt_model import SNOPTRepositioningDynamicsModel
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

    if name == "snopt":
        return SNOPTRepositioningDynamicsModel(active_arm, passive_arm, dt)

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

        pad = 0.0  # add pad to prevent contact forces
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
            if t < 0.25:
                return [1, -1]
            if t < 0.5:
                return [-1, 1]
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

        # Make consistent with Eric test.
        np.random.seed(1)

        def _torque_fn(t: float) -> list[float]:
            torque = (np.random.sample((6, 1)) - np.ones((6, 1)) * 0.5) * 20.0
            return list(torque.squeeze())

        return robot, human, _torque_fn, camera_kwargs

    raise NotImplementedError


def _main(scenario: str, dynamics: str, make_video: bool, video_dt: float) -> None:
    dt = 1 / 240
    T = 1000 * dt
    t = 0.0

    active_arm, passive_arm, torque_fn, camera_kwargs = _create_scenario(scenario)
    dynamics_model = _create_dynamics_model(dynamics, active_arm, passive_arm, dt)

    # Need to disable default velocity control to use torque control.
    for robot in [active_arm, passive_arm]:
        p.setJointMotorControlArray(
            robot.robot_id,
            robot.arm_joints,
            p.VELOCITY_CONTROL,
            forces=np.zeros(len(robot.arm_joints)),
            physicsClientId=robot.physics_client_id,
        )
        for joint in robot.arm_joints:
            # Turn on torque sensor.
            p.enableJointForceTorqueSensor(
                robot.robot_id, joint, 1, physicsClientId=robot.physics_client_id
            )

            # Remove any joint friction.
            p.changeDynamics(
                robot.robot_id,
                joint,
                jointDamping=0.0,
                anisotropicFriction=0.0,
                maxJointVelocity=5000,
                linearDamping=0.0,
                angularDamping=0.0,
                lateralFriction=0.0,
                spinningFriction=0.0,
                rollingFriction=0.0,
                contactStiffness=0.0,
                contactDamping=0.0,
                physicsClientId=robot.physics_client_id,
            )

            # Disable any possible collisions.
            p.setCollisionFilterGroupMask(
                robot.robot_id, joint, 0, 0, physicsClientId=robot.physics_client_id
            )

        # Let the simulation settle.
        for i in range(1000):
            p.stepSimulation(physicsClientId=robot.physics_client_id)

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
