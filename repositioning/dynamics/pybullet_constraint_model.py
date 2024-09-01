"""A model where PyBullet sets a constraint."""

import pybullet as p
from pybullet_helpers.geometry import multiply_poses

from .base_model import RepositioningDynamicsModel


class PybulletConstraintRepositioningDynamicsModel(RepositioningDynamicsModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        tf = multiply_poses(
            self._active_arm.get_end_effector_pose().invert(),
            self._passive_arm.get_end_effector_pose(),
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
        pybullet_dt = 1.0 / 240
        while t <= self._dt:
            p.stepSimulation(physicsClientId=self._active_arm.physics_client_id)
            t += pybullet_dt
