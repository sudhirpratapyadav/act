"""State-based scripted policy that solves PickPlaceCube.

Phase machine driven by current EEF position vs. cube/target. Outputs an OSC
delta-pose action: [dx, dy, dz, droll, dpitch, dyaw, gripper] in robosuite's
normalised [-1, 1] action space.

Used only for demo collection — the trained ACT policy gets just images +
proprio state at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# OSC_POSE position output range is [-0.05, 0.05] m per step. Dividing the
# desired delta by this scale and clipping to [-1, 1] gives the action.
OSC_POS_SCALE = 0.05

APPROACH_HEIGHT = 0.10   # above cube while approaching
GRASP_HEIGHT = 0.0       # eef site at cube center
LIFT_HEIGHT = 0.15
TRANSIT_HEIGHT = 0.15
PLACE_HEIGHT = 0.005     # cube bottom rests on table; eef at ~cube halfsize above target top
CUBE_HALFSIZE = 0.022

PHASES = ("approach", "descend", "close", "lift", "transit", "place_above",
          "place_down", "release", "retract", "done")


@dataclass
class ScriptState:
    phase: str = "approach"
    counter: int = 0
    initial_cube_pos: Optional[np.ndarray] = None  # cached at start
    extras: dict = field(default_factory=dict)


def _delta_action(eef_pos: np.ndarray, goal: np.ndarray, max_step=OSC_POS_SCALE) -> np.ndarray:
    """3-vector action in [-1, 1], normalised by `max_step`."""
    delta = goal - eef_pos
    return np.clip(delta / max_step, -1.0, 1.0)


def script_action(obs: dict, env, state: ScriptState) -> np.ndarray:
    """Return a 7-d OSC_POSE action for the current observation."""
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    cube_pos = np.asarray(obs["cube_pos"], dtype=np.float64)
    target_pos = np.asarray(env.target_pos, dtype=np.float64)

    if state.initial_cube_pos is None:
        state.initial_cube_pos = cube_pos.copy()

    action = np.zeros(7, dtype=np.float64)
    action[6] = -1.0  # gripper open by default

    if state.phase == "approach":
        goal = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + APPROACH_HEIGHT])
        action[:3] = _delta_action(eef_pos, goal)
        if np.linalg.norm(eef_pos - goal) < 0.012:
            state.phase = "descend"

    elif state.phase == "descend":
        goal = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + GRASP_HEIGHT])
        action[:3] = _delta_action(eef_pos, goal, max_step=0.025)
        if np.linalg.norm(eef_pos - goal) < 0.006:
            state.phase = "close"
            state.counter = 0

    elif state.phase == "close":
        action[6] = 1.0  # close
        state.counter += 1
        if state.counter >= 10:
            state.phase = "lift"

    elif state.phase == "lift":
        action[6] = 1.0
        goal = np.array(
            [
                state.initial_cube_pos[0],
                state.initial_cube_pos[1],
                state.initial_cube_pos[2] + LIFT_HEIGHT,
            ]
        )
        action[:3] = _delta_action(eef_pos, goal)
        if eef_pos[2] > goal[2] - 0.02:
            state.phase = "transit"

    elif state.phase == "transit":
        action[6] = 1.0
        goal = np.array(
            [target_pos[0], target_pos[1], target_pos[2] + TRANSIT_HEIGHT]
        )
        action[:3] = _delta_action(eef_pos, goal)
        if np.linalg.norm(eef_pos[:2] - goal[:2]) < 0.012:
            state.phase = "place_above"

    elif state.phase == "place_above":
        action[6] = 1.0
        goal = np.array(
            [target_pos[0], target_pos[1], target_pos[2] + 0.05]
        )
        action[:3] = _delta_action(eef_pos, goal, max_step=0.03)
        if abs(eef_pos[2] - goal[2]) < 0.01:
            state.phase = "place_down"

    elif state.phase == "place_down":
        action[6] = 1.0
        goal = np.array(
            [target_pos[0], target_pos[1], target_pos[2] + CUBE_HALFSIZE + PLACE_HEIGHT]
        )
        action[:3] = _delta_action(eef_pos, goal, max_step=0.015)
        if eef_pos[2] - goal[2] < 0.004:
            state.phase = "release"
            state.counter = 0

    elif state.phase == "release":
        action[6] = -1.0
        state.counter += 1
        if state.counter >= 12:
            state.phase = "retract"

    elif state.phase == "retract":
        action[6] = -1.0
        goal = np.array(
            [target_pos[0], target_pos[1], target_pos[2] + 0.20]
        )
        action[:3] = _delta_action(eef_pos, goal)
        if eef_pos[2] > goal[2] - 0.03:
            state.phase = "done"

    else:  # done
        action[6] = -1.0  # idle, gripper open

    return action.astype(np.float32)
