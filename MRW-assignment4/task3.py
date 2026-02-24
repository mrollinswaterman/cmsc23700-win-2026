import polyscope as ps
import polyscope.imgui as psim
import numpy as np

from skin import SkinMesh
from skeleton import Skeleton
from joint import Joint
import cube


"""
TODO -- Task 3: complete __init__ with your additional joints
"""


class Task3:
    def __init__(self):
        # Create empty skeleton
        self.skeleton = Skeleton()

        # TODO: Task-3
        # Create additional joints as required
        self.joint1 = Joint(None, (15, 0, 0), (0, 0, -1), -6.5, "Upper_Arm")
        self.joint2 = Joint(self.joint1, (-7, 0, 0), (0, -1, 0), -5.5, "Forearm")

        # join = (parent, pos, axis, length, name)
        self.joint3 = Joint(self.joint2, (-5.5, 0, 0), (1, 0, 0), -1.75, "Wrist")

        # fingies
        self.joint4 = Joint(
            self.joint3, (-2.35, 0, -0.1), (0, 0, 1), -1.5, "Middle Finger"
        )
        self.joint4.joint_angle = 18.0

        self.joint5 = Joint(
            self.joint3, (-2.25, -0.1, -0.75), (0, 0, 1), -1.25, "Index Finger"
        )
        self.joint5.joint_angle = 20.0

        self.joint6 = Joint(
            self.joint3, (-2.25, -0.1, 0.58), (0, 0, 1), -1.25, "Ring Finger"
        )
        self.joint6.joint_angle = 20.0

        self.joint7 = Joint(
            self.joint3, (-2.1, -0.3, 1.0), (0, 0, 1), -1.25, "Pinky Finger"
        )
        self.joint7.joint_angle = 30.0

        self.joint8 = Joint(
            self.joint3, (-1.0, -0.2, -1.0), (-0.5, -0.3, 1), -1.1, "Thumb"
        )
        self.joint8.joint_angle = 30.0

        # Add your joints to the skeleton here
        self.skeleton.add_joint(self.joint1)
        self.skeleton.add_joint(self.joint2)
        self.skeleton.add_joint(self.joint3)
        self.skeleton.add_joint(self.joint4)
        self.skeleton.add_joint(self.joint5)
        self.skeleton.add_joint(self.joint6)
        self.skeleton.add_joint(self.joint7)
        self.skeleton.add_joint(self.joint8)

        # Create skin mesh with skeleton
        self.skin_mesh = SkinMesh.create_arm_skinmesh(self.skeleton, "linear")

    # Render bones and mesh
    def render(self):
        joints = self.skeleton.joints
        positions, indices = (
            np.array(cube.Positions).reshape((-1, 3)),
            np.array(cube.Indices).reshape((-1, 3)),
        )
        init = 1
        options = [joint.name for joint in joints]
        joint_selected = options[0]
        ps_mesh = None

        def callback():
            nonlocal joints, init, options, joint_selected, ps_mesh
            change_flag = 0
            for i, joint in enumerate(joints):
                changed, joint.joint_angle = psim.SliderFloat(  # type: ignore
                    f"angle_{joint.name}", joint.joint_angle, v_min=0.0, v_max=360.0
                )  # type: ignore
                if changed:
                    change_flag = 1
                    self.set_joint_angle(i, joint.joint_angle)

            psim.PushItemWidth(200)  # type: ignore
            changed = psim.BeginCombo("Pick bone to show weight", joint_selected)  # type: ignore
            if changed:
                change_flag = 1
                for val in options:
                    _, selected = psim.Selectable(val, joint_selected == val)  # type: ignore
                    if selected:
                        joint_selected = val
                psim.EndCombo()  # type: ignore
            psim.PopItemWidth()  # type: ignore

            if change_flag or init:
                init = 0
                if ps_mesh is None:
                    ps_mesh = ps.register_surface_mesh(
                        "arm",
                        np.asarray(self.skin_mesh.transformed_positions).reshape(
                            (-1, 3)
                        ),
                        np.asarray(self.skin_mesh.indices).reshape((-1, 3)),
                        transparency=0.6,
                    )
                ps_mesh.update_vertex_positions(
                    np.asarray(self.skin_mesh.transformed_positions).reshape(-1, 3)
                )
                ps_mesh.add_scalar_quantity(
                    "weights",
                    np.asarray(self.skin_mesh.weights).reshape((-1, len(joints)))[
                        :, options.index(joint_selected)
                    ],
                    enabled=True,
                )
                for i, joint in enumerate(joints):
                    transformed_pos = positions.copy()
                    for i, p in enumerate(transformed_pos):
                        transformed_pos[i] = (joint.compute_model_matrix() @ [*p, 1])[
                            :-1
                        ]
                    ps.register_surface_mesh(
                        f"joint_{joint.name}", transformed_pos, indices
                    )

        ps.init()
        ps.set_user_callback(callback)
        ps.show()

    def set_joint_angle(self, joint_idx: int, joint_angle: float):
        self.skeleton.get_joint(joint_idx).joint_angle = joint_angle
        self.skin_mesh.update_skin()
