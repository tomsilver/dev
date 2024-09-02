import roboticstoolbox as rtb
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.gui import create_gui_connection
from roboticstoolbox.robot.Robot import Robot as RTBRobot
import spatialmath.base as base
import time
import numpy as np
from sympy import expand, collect, sympify, Array, solve, cos, sin, simplify
import pybullet as p
import dill

# physics_client_id = create_gui_connection()
# physics_client_id = p.connect(p.DIRECT)
# pybullet_robot = create_pybullet_robot("panda-limb-repo", physics_client_id)
# rtb_robot = RTBRobot.URDF(pybullet_robot.urdf_path())
# rtb_robot._symbolic = True
# print(rtb_robot)

# rtb_robot = rtb.models.DH.Panda()
# q0 = [0.0] * 7

# q0 = pybullet_robot.get_joint_positions()

# rtb_robot = rtb.models.DH.Panda()
# rtb_robot._symbolic = True
# robot_name = "panda_dh_7d0f"

rtb_robot = rtb.models.DH.TwoLink()
rtb_robot._symbolic = True
robot_name = "two_link"

# disabling gravity
rtb_robot.gravity = [0.0, 0.0, 0.0]

# rtb_robot = rtb.models.DH.Puma560(symbolic=True)
# robot_name = "panda"

q0 = base.sym.symbol(f"q_:{rtb_robot.n}")
qd0 = base.sym.symbol(f"qd_:{rtb_robot.n}")

# Forward kinematics
ee = rtb_robot.fkine(q0)
# TODO handle rotations

eefilename = f"{robot_name}_ee.p"
with open(eefilename, "wb") as f:
    dill.dump([ee.x, ee.y, ee.z], f)

M = rtb_robot.inertia(q0)
C = rtb_robot.coriolis(q0, qd0)
g = rtb_robot.gravload(q0)


Mfilename = f"{robot_name}_M.p"
Cfilename = f"{robot_name}_C.p"
gfilename = f"{robot_name}_g.p"
equation_filename = f"{robot_name}_equation.p"
simplified_equation_filename = f"{robot_name}_equation_simplified.p"

with open(Mfilename, "wb") as f:
    dill.dump(M, f)
with open(Cfilename, "wb") as f:
    dill.dump(C, f)
with open(gfilename, "wb") as f:
    dill.dump(g, f)

with open(Mfilename, "rb") as f:
    M = dill.load(f)
with open(Cfilename, "rb") as f:
    C = dill.load(f)
with open(gfilename, "rb") as f:
    g = dill.load(f)

qdd0 = base.sym.symbol(f"qdd_:{rtb_robot.n}")
tau0 = base.sym.symbol(f"tau_:{rtb_robot.n}")

# Manipulator equation
equation =  M @ qdd0 + C @ qd0 - (tau0 + g)

with open(equation_filename, "wb") as f:
    dill.dump(equation, f)

with open(equation_filename, "rb") as f:
    equation = dill.load(f)

# Simplify (TOO SLOW)
# common_terms = []
# for v in [q0, qd0, tau0]:
#     for i in range(rtb_robot.n):
#         common_terms.append(sin(v[i]))
#         common_terms.append(cos(v[i]))

# simplified_equation = [collect(l, common_terms) for l in equation]
simplified_equation = [simplify(l) for l in equation]
for eq in simplified_equation:
    print(eq)
# with open(simplified_equation_filename, "wb") as f:
#     dill.dump(simplified_equation, f)

# # Evaluate to get qdd0 given everything else.
# subs = {}
# for v in [q0, qd0, tau0]:
#     for i in range(rtb_robot.n):
#         subs[v[i]] = 0.

# exprs = [l.subs(list(subs.items())) for l in equation]

# # Solve for qdd0.
# solution = solve(exprs, qdd0)

# traj = rtb_robot.nofriction().fdyn(1, q0, dt=0.001)

# for q in traj.q:
#     pybullet_robot.set_joints(q)
#     time.sleep(0.01)

