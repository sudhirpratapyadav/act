"""Run the scripted policy and stream successful demos into a LeRobot dataset.

Output: data/lerobot/pick_place_cube/  (LeRobotDataset, image-backed, 84×84,
two cameras, fps=20)
Failed rollouts are simply skipped — we keep going until `--num-demos`
successful episodes are recorded.

Run:
    uv run python scripts/07_pickplace_collect.py --num-demos 100
"""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import robosuite
from robosuite.controllers import load_composite_controller_config

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripts.envs.pick_place_cube import PickPlaceCube, register
from scripts.envs.scripted_policy import ScriptState, script_action


REPO_ID = "local/pick_place_cube"
TASK_TEXT = "pick the red cube and place it on the green target"
FPS = 20
IMG_HW = 84
HORIZON = 220  # scripted episodes finish in ~80–110 steps; pad a bit
CAMERAS = ["agentview", "robot0_eye_in_hand"]
CAMERA_KEYS = {
    "agentview": "observation.images.agentview",
    "robot0_eye_in_hand": "observation.images.wrist",
}
OUT_ROOT = ROOT / "data" / "lerobot" / "pick_place_cube"


def proprio_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


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
        camera_names=CAMERAS,
        camera_heights=IMG_HW,
        camera_widths=IMG_HW,
        horizon=HORIZON,
        control_freq=FPS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": (9,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    for cam in CAMERAS:
        features[CAMERA_KEYS[cam]] = {
            "dtype": "image",
            "shape": (IMG_HW, IMG_HW, 3),
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, features=features,
        root=OUT_ROOT, robot_type="panda", use_videos=False,
    )

    env = make_env()
    np.random.seed(args.seed)

    rng = np.random.RandomState(args.seed)
    successes = 0
    attempts = 0
    pbar = tqdm(total=args.num_demos, desc="successful demos")
    while successes < args.num_demos:
        # Use a fresh seed for each attempt to keep variability deterministic.
        seed = int(rng.randint(0, 2**31 - 1))
        np.random.seed(seed)
        obs = env.reset()
        attempts += 1

        state = ScriptState()
        states_buf, action_buf, agent_buf, wrist_buf = [], [], [], []

        success = False
        for t in range(HORIZON):
            action = script_action(obs, env, state)
            states_buf.append(proprio_state(obs))
            action_buf.append(action.astype(np.float32))
            agent_buf.append(obs["agentview_image"])
            wrist_buf.append(obs["robot0_eye_in_hand_image"])

            obs, _r, _done, _info = env.step(action)
            if env._check_success() and state.phase in ("retract", "done"):
                # one final frame after success so the dataset includes the released state
                states_buf.append(proprio_state(obs))
                action_buf.append(action.astype(np.float32))  # idle action
                agent_buf.append(obs["agentview_image"])
                wrist_buf.append(obs["robot0_eye_in_hand_image"])
                success = True
                break

        if not success:
            continue

        for s, a, ag, wr in zip(states_buf, action_buf, agent_buf, wrist_buf):
            dataset.add_frame(
                {
                    "observation.state": s,
                    "action": a,
                    CAMERA_KEYS["agentview"]: ag,
                    CAMERA_KEYS["robot0_eye_in_hand"]: wr,
                },
                task=TASK_TEXT,
            )
        dataset.save_episode()
        successes += 1
        pbar.update(1)

    pbar.close()
    print(
        f"\nrecorded {successes} demos in {attempts} attempts "
        f"(success rate {successes/attempts:.1%})"
    )
    print(f"dataset at {OUT_ROOT}: {dataset.num_episodes} episodes / "
          f"{dataset.num_frames} frames")


if __name__ == "__main__":
    main()
