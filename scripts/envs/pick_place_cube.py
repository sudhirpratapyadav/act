"""Cube pick-and-place env.

A red cube spawns on the left half of the table; a flat green visual square
marks a fixed target on the right half. The agent must place the cube on the
target.

Implemented as a subclass of robosuite.environments.manipulation.lift.Lift —
we reuse Lift's cube + observables, narrow the spawn to the left half, and
inject the target as a static visual geom in the arena worldbody (no joint,
so MuJoCo doesn't require positive inertia for it).

Success condition (`_check_success`):
    cube center within `success_radius` of the target XY  AND
    cube near table height                                AND
    gripper has released the cube
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler


# Cube spawn box on the left side of the table.
CUBE_X_RANGE = (-0.08, -0.04)
CUBE_Y_RANGE = (-0.04, 0.04)
# Target spawn box on the right side of the table — randomized each reset.
TARGET_X_RANGE = (0.08, 0.18)
TARGET_Y_RANGE = (-0.08, 0.08)
TARGET_HALFSIZE = 0.04   # 8 cm × 8 cm visual square
SUCCESS_RADIUS = 0.045   # cube counts as placed within this XY radius
TABLE_TOP_Z = 0.82       # default table_offset[2] = 0.8, top is +0.02

TARGET_GEOM_NAME = "target_zone"


class PickPlaceCube(Lift):
    """Pick a red cube and place it on a randomly placed green target zone."""

    def __init__(self, *args,
                 target_x_range=TARGET_X_RANGE,
                 target_y_range=TARGET_Y_RANGE,
                 success_radius=SUCCESS_RADIUS, **kwargs):
        self._target_x_range = tuple(float(v) for v in target_x_range)
        self._target_y_range = tuple(float(v) for v in target_y_range)
        self._success_radius = float(success_radius)
        # Concrete target offset for this episode — set at reset.
        self._target_offset_xy = np.array(
            [
                0.5 * (self._target_x_range[0] + self._target_x_range[1]),
                0.5 * (self._target_y_range[0] + self._target_y_range[1]),
            ],
            dtype=np.float64,
        )
        super().__init__(*args, **kwargs)

    @property
    def target_pos(self) -> np.ndarray:
        return np.array(
            [
                self.table_offset[0] + self._target_offset_xy[0],
                self.table_offset[1] + self._target_offset_xy[1],
                self.table_offset[2] + 0.02,  # table top
            ],
            dtype=np.float64,
        )

    def _sample_target_offset(self) -> np.ndarray:
        # Uniform sampler driven by the env's current numpy RNG, so callers can
        # still seed reproducibly via np.random.seed before reset().
        x = np.random.uniform(*self._target_x_range)
        y = np.random.uniform(*self._target_y_range)
        return np.array([x, y], dtype=np.float64)

    def _load_model(self):
        # Mirror Lift._load_model but: (a) narrow cube spawn to the left half,
        # (b) inject a static green geom in the arena worldbody as the target.
        super(Lift, self)._load_model()  # ManipulationEnv._load_model

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Static, non-colliding visual square — initial XYZ from the current
        # `_target_offset_xy`. We move it each reset via `sim.model.geom_pos`.
        target_world_xyz = (
            float(self.table_offset[0] + self._target_offset_xy[0]),
            float(self.table_offset[1] + self._target_offset_xy[1]),
            float(self.table_offset[2] + 0.0205),  # tiny epsilon above table top
        )
        target_geom = ET.Element(
            "geom",
            attrib={
                "name": TARGET_GEOM_NAME,
                "type": "box",
                "size": f"{TARGET_HALFSIZE} {TARGET_HALFSIZE} 0.0015",
                "pos": f"{target_world_xyz[0]} {target_world_xyz[1]} {target_world_xyz[2]}",
                "rgba": "0.1 0.85 0.2 0.9",
                "contype": "0",      # no collisions
                "conaffinity": "0",
                "group": "1",
            },
        )
        mujoco_arena.worldbody.append(target_geom)

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
            tex_attrib=tex_attrib, mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cube,
            x_range=list(CUBE_X_RANGE),
            y_range=list(CUBE_Y_RANGE),
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        super()._setup_references()
        self._target_geom_id = self.sim.model.geom_name2id(TARGET_GEOM_NAME)

    def _reset_internal(self):
        super()._reset_internal()
        # Sample a new target XY and move the static visual geom to it.
        self._target_offset_xy = self._sample_target_offset()
        new_pos = np.array(
            [
                self.table_offset[0] + self._target_offset_xy[0],
                self.table_offset[1] + self._target_offset_xy[1],
                self.table_offset[2] + 0.0205,
            ],
            dtype=np.float64,
        )
        self.sim.model.geom_pos[self._target_geom_id] = new_pos
        self.sim.forward()

    def _check_success(self):
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        xy_dist = float(np.linalg.norm(cube_pos[:2] - self.target_pos[:2]))
        on_table = cube_pos[2] < TABLE_TOP_Z + 0.04
        gripped = self._check_grasp(
            gripper=self.robots[0].gripper, object_geoms=self.cube
        )
        return (xy_dist < self._success_radius) and on_table and (not gripped)


def register():
    """Register the env class with robosuite so robosuite.make("PickPlaceCube") works."""
    from robosuite.environments.base import REGISTERED_ENVS
    REGISTERED_ENVS[PickPlaceCube.__name__] = PickPlaceCube
