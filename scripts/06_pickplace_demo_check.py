"""Run the scripted policy once on PickPlaceCube and save an mp4 for review."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import imageio.v2 as imageio
import robosuite
from robosuite.controllers import load_composite_controller_config

from scripts.envs.pick_place_cube import PickPlaceCube, register
from scripts.envs.scripted_policy import ScriptState, script_action

OUT_DIR = ROOT / "videos" / "verification"
HORIZON = 250
SEED = 0
RECORD_HW = 256


def make_env():
    register()
    config = load_composite_controller_config(controller="BASIC", robot="Panda")
    return robosuite.make(
        "PickPlaceCube",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        ignore_done=True,
        reward_shaping=False,
        camera_names=["agentview", "robot0_eye_in_hand", "frontview"],
        camera_heights=[84, 84, RECORD_HW],
        camera_widths=[84, 84, RECORD_HW],
        horizon=HORIZON,
        control_freq=20,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env()
    np.random.seed(SEED)

    obs = env.reset()
    state = ScriptState()
    frames = [np.flipud(obs["frontview_image"]).copy()]

    success = False
    final_phase = state.phase
    for t in range(HORIZON):
        action = script_action(obs, env, state)
        obs, _r, _done, _info = env.step(action)
        frames.append(np.flipud(obs["frontview_image"]).copy())
        final_phase = state.phase
        if env._check_success() and state.phase in ("retract", "done"):
            success = True
            break

    out = OUT_DIR / "pickplace_scripted.mp4"
    with imageio.get_writer(out, fps=20, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)

    print(
        f"saved {out}  ({len(frames)} frames)  "
        f"final_phase={final_phase}  success={success}  "
        f"final_cube_xy={env.sim.data.body_xpos[env.cube_body_id][:2]}"
    )


if __name__ == "__main__":
    main()
