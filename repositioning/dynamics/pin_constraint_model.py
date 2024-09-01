"""A model where Pinocchio sets a constraint."""

from pybullet_helpers.geometry import multiply_poses
import numpy as np
import pinocchio as pin

from .base_model import RepositioningDynamicsModel
from pinocchio.visualize import MeshcatVisualizer as Visualizer
import hppfcl as fcl


class PinConstraintRepositioningDynamicsModel(RepositioningDynamicsModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Load robots in pinnochio.
        active_model, _, active_visual_model = pin.buildModelsFromUrdf(
            str(self._active_arm.urdf_path()),
            package_dirs=str(self._active_arm.urdf_path().parent),
        )
        passive_model, _, passive_visual_model = pin.buildModelsFromUrdf(
            str(self._passive_arm.urdf_path()),
            package_dirs=str(self._passive_arm.urdf_path().parent),
        )

        # TODO
        transform = pin.SE3.Identity()

        joint_index = active_model.addJoint(0, pin.JointModelFixed(), transform, 'fixed_joint')
        active_model.model.addBody(active_model.model.joints[joint_index], passive_model.model.inertias[1], passive_model.model.frames[1])
        active_model.data = active_model.model.createData()


        # # build model from scratch
        # model2 = pin.Model()
        # model2.name = "pendulum"
        # geom_model = pin.GeometryModel()

        # parent_id = 0
        # joint_placement = pin.SE3.Identity()
        # body_mass = 1.0
        # body_radius = 1e-2

        # joint_name = "joint_spherical"
        # joint_id = model2.addJoint(
        #     parent_id, pin.JointModelSpherical(), joint_placement, joint_name
        # )

        # body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        # body_placement = joint_placement.copy()
        # body_placement.translation[2] = 0.1
        # model2.appendBodyToJoint(joint_id, body_inertia, body_placement)

        # geom1_name = "ball"
        # shape1 = fcl.Sphere(body_radius)
        # geom1_obj = pin.GeometryObject(geom1_name, joint_id, body_placement, shape1)
        # geom1_obj.meshColor = np.ones((4))
        # geom_model.addGeometryObject(geom1_obj)

        # geom2_name = "bar"
        # shape2 = fcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
        # shape2_placement = body_placement.copy()
        # shape2_placement.translation[2] /= 2.0

        # geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2_placement, shape2)
        # geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        # geom_model.addGeometryObject(geom2_obj)

        # visual_model2 = geom_model

        # # join the two models, append pendulum to end effector
        # frame_id_end_effector = active_model.getFrameId(self._active_arm.end_effector_name)
        # model, visual_model = pin.appendModel(
        #     active_model,
        #     model2,
        #     active_visual_model,
        #     visual_model2,
        #     frame_id_end_effector,
        #     pin.SE3.Identity(),
        # )

        # try:
        #     viz = Visualizer(model, visual_model, visual_model)
        #     viz.initViewer(open=True)
        # except ImportError as err:
        #     print(
        #         "Error while initializing the viewer. It seems you should install Python meshcat"
        #     )
        #     print(err)
        #     sys.exit(0)

        # # Load the robot in the viewer.
        # viz.loadViewerModel()

        # # Display a random robot configuration.
        # model.lowerPositionLimit.fill(-np.pi / 2)
        # model.upperPositionLimit.fill(np.pi / 2)
        # q = pin.randomConfiguration(model)
        # viz.display(q)

        # Create constraint.
        tf = multiply_poses(
            self._passive_arm.get_end_effector_pose().invert(),
            self._active_arm.get_end_effector_pose(),
        )





    def step(self, torque: list[float]) -> None:
        import ipdb; ipdb.set_trace()