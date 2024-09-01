"""A model where PyBullet sets a constraint."""

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import multiply_poses

from .base_model import RepositioningDynamicsModel


class PybulletConstraintRepositioningDynamicsModel(RepositioningDynamicsModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Create constraint.
        tf = multiply_poses(
            self._passive_arm.get_end_effector_pose().invert(),
            self._active_arm.get_end_effector_pose(),
        )
        p.createConstraint(
            self._active_arm.robot_id,
            self._active_arm.tool_link_id,
            self._passive_arm.robot_id,
            self._passive_arm.tool_link_id,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=tf.position,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=tf.orientation,
            physicsClientId=self._active_arm.physics_client_id,
        )
        # Need to disable default velocity control to use torque control.
        for robot in [self._active_arm, self._passive_arm]:
            p.setJointMotorControlArray(
                robot.robot_id,
                robot.arm_joints,
                p.VELOCITY_CONTROL,
                forces=np.zeros(len(robot.arm_joints)),
                physicsClientId=robot.physics_client_id,
            )

    def step(self, torque: list[float]) -> None:
        # TODO: move this into pybullet helpers.
        p.setJointMotorControlArray(
            self._active_arm.robot_id,
            self._active_arm.arm_joints,
            p.TORQUE_CONTROL,
            forces=torque,
            physicsClientId=self._active_arm.physics_client_id,
        )
        t = 0.0
        pybullet_dt = 1 / 240  # default pybullet dt
        while t <= self._dt:
            p.stepSimulation(physicsClientId=self._active_arm.physics_client_id)
            t += pybullet_dt
