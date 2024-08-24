import numpy as np
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, RigidTransform, MathematicalProgram

####################################
# Minimal broken code
####################################
dt = 1e-2

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
urdf = "models/panda_fr3/urdf/panda_fr3.urdf"
arm = Parser(plant).AddModels(urdf)[0]
X_robot = RigidTransform()
X_robot.set_translation([0, 0, 0.015])
plant.WeldFrames(plant.world_frame(),
                    plant.GetFrameByName("panda_link0", arm),
                    X_robot)
plant.Finalize()
diagram = builder.Build()

autodiff_diagram = diagram.ToAutoDiffXd()
autodiff_plant = autodiff_diagram.GetSubsystemByName("plant")
autodiff_diagram_context = autodiff_diagram.CreateDefaultContext()
autodiff_plant_context = autodiff_diagram.GetMutableSubsystemContext(autodiff_plant, autodiff_diagram_context)

prog = MathematicalProgram()
q0 = prog.NewContinuousVariables(7, "q_0")
autodiff_plant.SetPositions(autodiff_plant_context, q0)

import ipdb; ipdb.set_trace()
