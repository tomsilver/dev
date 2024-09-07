"""A math repositioning dynamics model."""

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import matrix_from_quat
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

from .base_model import RepositioningDynamicsModel
from ..structs import RepositioningState, JointTorques


class MathRepositioningDynamicsModel(RepositioningDynamicsModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_to_passive_ee_twist = self._get_active_to_passive_ee_twist(
            self.active_arm, self.passive_arm
        )

    def reset(self, state: RepositioningState) -> None:
        self.active_arm.set_joints(state.active_positions, state.active_velocities)
        self.passive_arm.set_joints(state.passive_positions, state.passive_velocities)

    def get_state(self) -> RepositioningState:
        active_positions = self.active_arm.get_joint_positions()
        active_velocities = self.active_arm.get_joint_velocities()
        passive_positions = self.passive_arm.get_joint_positions()
        passive_velocities = self.passive_arm.get_joint_velocities()
        return RepositioningState(active_positions, active_velocities, passive_positions, passive_velocities)

    def step(self, torque: JointTorques) -> None:
        pos_r = np.array(self.active_arm.get_joint_positions())
        pos_h = np.array(self.passive_arm.get_joint_positions())
        vel_r = np.array(self.active_arm.get_joint_velocities())
        vel_h = np.array(self.passive_arm.get_joint_velocities())
        R = self._active_to_passive_ee_twist

        Jr = self._calculate_jacobian(self.active_arm)
        Jh = self._calculate_jacobian(self.passive_arm)
        Jhinv = np.linalg.pinv(Jh)

        Mr = self._calculate_mass_matrix(self.active_arm)
        Mh = self._calculate_mass_matrix(self.passive_arm)

        Nr = self._calculate_N_vector(self.active_arm)
        Nh = self._calculate_N_vector(self.passive_arm)

        acc_r = np.linalg.pinv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
            (Jhinv @ R @ Jr).T
            @ (
                Mh * (1 / self.dt) @ (Jhinv @ R @ Jr) @ vel_r
                - Mh * (1 / self.dt) @ vel_h
                + Nh
            )
            + Nr
            - np.array(torque)
        )

        new_vel_r = vel_r + acc_r * self.dt
        r_lin_vel = Jr @ new_vel_r
        h_lin_vel = R @ r_lin_vel
        new_vel_h = Jhinv @ h_lin_vel

        acc_h = (new_vel_h - vel_h) / self.dt

        vel_r = vel_r + acc_r * self.dt
        vel_h = vel_h + acc_h * self.dt

        pos_r = pos_r + vel_r * self.dt
        pos_h = pos_h + vel_h * self.dt

        self.active_arm.set_joints(list(pos_r), joint_velocities=list(vel_r))
        self.passive_arm.set_joints(list(pos_h), joint_velocities=list(vel_h))

    @staticmethod
    def _calculate_jacobian(robot: SingleArmPyBulletRobot) -> NDArray:
        joint_positions = robot.get_joint_positions()
        jac_t, jac_r = p.calculateJacobian(
            robot.robot_id,
            robot.tool_link_id,
            [0, 0, 0],
            joint_positions,
            [0.0] * len(joint_positions),
            [0.0] * len(joint_positions),
            physicsClientId=robot.physics_client_id,
        )
        return np.concatenate((np.array(jac_t), np.array(jac_r)), axis=0)

    @staticmethod
    def _calculate_mass_matrix(robot: SingleArmPyBulletRobot) -> NDArray:
        mass_matrix = p.calculateMassMatrix(
            robot.robot_id,
            robot.get_joint_positions(),
            physicsClientId=robot.physics_client_id,
        )
        return np.array(mass_matrix)

    @staticmethod
    def _calculate_N_vector(robot: SingleArmPyBulletRobot) -> NDArray:
        joint_positions = robot.get_joint_positions()
        joint_velocities = robot.get_joint_velocities()
        joint_accel = [0.0] * len(joint_positions)
        n_vector = p.calculateInverseDynamics(
            robot.robot_id,
            joint_positions,
            joint_velocities,
            joint_accel,
            physicsClientId=robot.physics_client_id,
        )
        return np.array(n_vector)

    @staticmethod
    def _get_active_to_passive_ee_twist(
        active_arm: SingleArmPyBulletRobot, passive_arm: SingleArmPyBulletRobot
    ) -> NDArray:
        active_ee_orn = active_arm._base_pose.orientation
        passive_ee_orn = passive_arm._base_pose.orientation
        active_to_passive_ee = matrix_from_quat(passive_ee_orn).T @ matrix_from_quat(
            active_ee_orn
        )
        active_to_passive_ee_twist = np.eye(6)
        active_to_passive_ee_twist[:3, :3] = active_to_passive_ee
        active_to_passive_ee_twist[3:, 3:] = active_to_passive_ee
        return active_to_passive_ee_twist
