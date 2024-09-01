"""A model that uses constrained nonlinear optimization with SNOPT."""

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import matrix_from_quat, multiply_poses, Pose
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pydrake.all import (MathematicalProgram, SnoptSolver, SolutionResult,
                         Expression, Variable, eq, le, sin, cos)
from spatialmath import SE3
import dill
from pathlib import Path

from .base_model import RepositioningDynamicsModel


class SNOPTRepositioningDynamicsModel(RepositioningDynamicsModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # TODO generalize
        assert self._active_arm.get_name() == self._passive_arm.get_name() == "two-link", "TODO"
        with open(Path(".").resolve().parent / "symbolic-dynamics" / "two_link_equation.p", "rb") as f:
            self._sympy_active_manip_equation = dill.load(f)
        self._sympy_passive_manip_equation = self._sympy_active_manip_equation
        with open(Path(".").resolve().parent / "symbolic-dynamics" / "two_link_ee.p", "rb") as f:
            self._sympy_active_fk_equation = dill.load(f)
        self._sympy_passive_fk_equation = self._sympy_active_fk_equation

        self._rng = np.random.default_rng(0)

        # TODO: incorporate robot base poses in FK constraint.
    
        from pybullet_helpers.gui import visualize_pose

        # active_q = self._active_arm.get_joint_positions()
        # base_to_ee = []
        # for i in range(len(self._sympy_active_fk_equation)):
        #     line = str(self._sympy_active_fk_equation[i])
        #     for j in range(2):
        #         line = line.replace(f"q_{j}", f"{active_q[j]}")
        #     result = eval(line)
        #     base_to_ee.append(result)
        # world_to_base = SE3.Rt(matrix_from_quat(self._active_arm._base_pose.orientation), self._active_arm._base_pose.position)
        # active_world_to_ee = (world_to_base * base_to_ee).squeeze()
        # world_to_base2 = self._active_arm._base_pose
        # world_to_ee2 = multiply_poses(self._active_arm._base_pose, Pose(base_to_ee))

        # passive_q = self._passive_arm.get_joint_positions()
        # base_to_ee = []
        # for i in range(len(self._sympy_passive_fk_equation)):
        #     line = str(self._sympy_passive_fk_equation[i])
        #     for j in range(2):
        #         line = line.replace(f"q_{j}", f"{passive_q[j]}")
        #     result = eval(line)
        #     base_to_ee.append(result)
        # world_to_base = SE3.Rt(matrix_from_quat(self._passive_arm._base_pose.orientation), self._passive_arm._base_pose.position)
        # passive_world_to_ee = (world_to_base * base_to_ee).squeeze()

        # world_to_base2 = self._passive_arm._base_pose
        # world_to_ee2 = multiply_poses(self._passive_arm._base_pose, Pose(base_to_ee))

        # visualize_pose(Pose(world_to_ee), self._active_arm.physics_client_id)
        # while True:
        #     p.stepSimulation()
                  

    def step(self, torque: list[float]) -> None:
        # TODO: cache program between steps
        # TODO: warm start from previous solutions

        program = MathematicalProgram()

        active_dof = len(self._active_arm.arm_joints)
        passive_dof = len(self._passive_arm.arm_joints)

        # torque = program.NewContinuousVariables(active_dof, "torque")
        # active_q = program.NewContinuousVariables(active_dof, "active_q")
        # active_qd = program.NewContinuousVariables(active_dof, "active_qd")

        active_q = self._active_arm.get_joint_positions()
        active_qd = self._active_arm.get_joint_velocities()
        active_qdd = program.NewContinuousVariables(active_dof, "active_qdd")
        next_active_q = program.NewContinuousVariables(active_dof, "next_active_q")
        next_active_qd = program.NewContinuousVariables(active_dof, "next_active_qd")
        next_active_ee = program.NewContinuousVariables(3, "next_active_ee")

        # passive_q = program.NewContinuousVariables(passive_dof, "passive_q")
        # passive_qd = program.NewContinuousVariables(passive_dof, "passive_qd")

        passive_q = self._passive_arm.get_joint_positions()
        passive_qd = self._passive_arm.get_joint_velocities()
        passive_qdd = program.NewContinuousVariables(passive_dof, "passive_qdd")
        passive_torque = program.NewContinuousVariables(passive_dof, "passive_torque")
        program.SetInitialGuess(passive_torque, self._rng.uniform(-10, 10, size=len(passive_torque)))

        next_passive_q = program.NewContinuousVariables(passive_dof, "next_passive_q")
        next_passive_qd = program.NewContinuousVariables(passive_dof, "next_passive_qd")
        next_passive_ee = program.NewContinuousVariables(3, "next_passive_ee")

        self._add_manipulator_equation_constraint(program, torque, passive_torque, active_q, active_qd, active_qdd,
                                                  passive_q, passive_qd, passive_qdd,
                                                  next_active_q, next_active_qd,
                                                  next_passive_q, next_passive_qd,
                                                  next_active_ee, next_passive_ee)
        
        solver = SnoptSolver()
        result = solver.Solve(program)
        assert result.is_success()
        print(result.get_optimal_cost())
        print(result.GetSolution(next_passive_ee))
        # print(result.GetSolution(passive_torque))
        print()

        self._active_arm.set_joints(result.GetSolution(next_active_q), joint_velocities=result.GetSolution(next_active_qd))
        self._passive_arm.set_joints(result.GetSolution(next_passive_q), joint_velocities=result.GetSolution(next_passive_qd))
        
    def _add_manipulator_equation_constraint(self, program, active_torque, passive_torque, active_q, active_qd, active_qdd,
                                                  passive_q, passive_qd, passive_qdd,
                                                  next_active_q, next_active_qd,
                                                  next_passive_q, next_passive_qd,
                                                  next_active_ee, next_passive_ee) -> None:
        # TODO make general and less disgusting

        # Add manipulator equation constraint for active arm.
        drake_active_manip = []
        for i in range(len(active_q)):
            line = str(self._sympy_active_manip_equation[i])
            for j in range(len(active_q)):
                line = line.replace(f"q_{j}", f"active_q[{j}]")
                line = line.replace(f"qd_{j}", f"active_qd[{j}]")
                line = line.replace(f"qdd_{j}", f"active_qdd[{j}]")
                line = line.replace(f"tau_{j}", f"active_torque[{j}]")
            drake_expression = eval(line)
            drake_active_manip.append(drake_expression)
        program.AddConstraint(eq(np.array(drake_active_manip), np.zeros(len(active_q))))

        # Add manipulator equation constraint for passive arm.
        drake_passive_manip = []
        for i in range(len(passive_q)):
            line = str(self._sympy_passive_manip_equation[i])
            for j in range(len(passive_q)):
                line = line.replace(f"q_{j}", f"passive_q[{j}]")
                line = line.replace(f"qd_{j}", f"passive_qd[{j}]")
                line = line.replace(f"qdd_{j}", f"passive_qdd[{j}]")
                line = line.replace(f"tau_{j}", f"passive_torque[{j}]")
            drake_expression = eval(line)
            drake_passive_manip.append(drake_expression)
        program.AddConstraint(eq(np.array(drake_passive_manip), np.zeros(len(passive_q))))

        # Active forward kinematics.
        drake_active_fk = []
        for i in range(3):
            line = str(self._sympy_active_fk_equation[i])
            for j in range(len(active_q)):
                line = line.replace(f"q_{j}", f"next_active_q[{j}]")
            drake_expression = eval(line)
            drake_active_fk.append(drake_expression)
        se3 = SE3.Rt(matrix_from_quat(self._active_arm._base_pose.orientation), self._active_arm._base_pose.position)
        transformed_next_active_ee = (se3.inv() * next_active_ee).squeeze()
        program.AddConstraint(eq(transformed_next_active_ee, drake_active_fk))

        # Passive forward kinematics.
        drake_passive_fk = []
        for i in range(3):
            line = str(self._sympy_passive_fk_equation[i])
            for j in range(len(passive_q)):
                line = line.replace(f"q_{j}", f"next_passive_q[{j}]")
            drake_expression = eval(line)
            drake_passive_fk.append(drake_expression)
        se3 = SE3.Rt(matrix_from_quat(self._passive_arm._base_pose.orientation), self._passive_arm._base_pose.position)
        transformed_next_passive_ee = (se3.inv() * next_passive_ee).squeeze()
        program.AddConstraint(eq(transformed_next_passive_ee, drake_passive_fk))
            
        # TODO handle orientation!
        drake_ee = []
        for i in range(3):
            dist = (next_active_ee[i] - next_passive_ee[i])**2
            drake_ee.append(dist)
        # program.AddConstraint(le(np.array(drake_ee), 1e-2 * np.ones(len(drake_ee))))
        program.AddCost(sum(drake_ee))

        # Integrate.
        program.AddConstraint(eq(next_active_qd, active_qd + active_qdd * self._dt))
        program.AddConstraint(eq(next_active_q, active_q + next_active_qd * self._dt))
        # NOTE: commenting this next line out helps...
        program.AddConstraint(eq(next_passive_qd, passive_qd + passive_qdd * self._dt))
        program.AddConstraint(eq(next_passive_q, passive_q + next_passive_qd * self._dt))

