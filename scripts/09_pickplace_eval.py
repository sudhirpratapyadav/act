"""Roll out trained ACT in PickPlaceCube and report success rate.

Saves a few mp4 rollouts to videos/eval_pickplace/ and a JSON summary.

Usage:
    uv run python scripts/09_pickplace_eval.py
    uv run python scripts/09_pickplace_eval.py --ckpt step_0030000 --num-episodes 100
"""
import argparse
import json
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

import robosuite
from robosuite.controllers import load_composite_controller_config

from lerobot.policies.act.modeling_act import ACTPolicy

from scripts.envs.pick_place_cube import PickPlaceCube, register

CKPT_ROOT = ROOT / "checkpoints" / "act_pickplace"
VIDEO_DIR = ROOT / "videos" / "eval_pickplace"

CAMERAS = ["agentview", "robot0_eye_in_hand"]
CAMERA_KEYS = {
    "agentview": "observation.images.agentview",
    "robot0_eye_in_hand": "observation.images.wrist",
}
TRAIN_IMG_HW = 84
RECORD_IMG_HW = 256
RECORD_CAM = "frontview"
ROLLOUT_HORIZON = 250
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def proprio_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def to_policy_batch(obs: dict) -> dict:
    batch = {
        "observation.state": torch.from_numpy(proprio_state(obs)).float().unsqueeze(0).to(DEVICE),
    }
    for cam, lerobot_key in CAMERA_KEYS.items():
        img = obs[f"{cam}_image"]
        t = torch.from_numpy(img).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        batch[lerobot_key] = t
    return batch


def make_env():
    register()
    config = load_composite_controller_config(controller="BASIC", robot="Panda")
    cam_names = list(CAMERAS) + [RECORD_CAM]
    cam_h = [TRAIN_IMG_HW] * len(CAMERAS) + [RECORD_IMG_HW]
    cam_w = [TRAIN_IMG_HW] * len(CAMERAS) + [RECORD_IMG_HW]
    return robosuite.make(
        "PickPlaceCube",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        ignore_done=True,
        camera_names=cam_names,
        camera_heights=cam_h,
        camera_widths=cam_w,
        horizon=ROLLOUT_HORIZON,
        control_freq=20,
    )


def render_record_frame(obs):
    return np.flipud(obs[f"{RECORD_CAM}_image"]).copy()


def pick_checkpoint(arg_ckpt: str | None) -> Path:
    if arg_ckpt:
        candidate = CKPT_ROOT / arg_ckpt
        if not candidate.exists():
            candidate = Path(arg_ckpt)
        return candidate
    last = CKPT_ROOT / "last"
    if last.exists():
        return last
    steps = sorted(CKPT_ROOT.glob("step_*"))
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {CKPT_ROOT}")
    return steps[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--num-videos", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=ROLLOUT_HORIZON)
    args = parser.parse_args()

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = pick_checkpoint(args.ckpt)
    print(f"loading policy from {ckpt_path}")
    policy = ACTPolicy.from_pretrained(ckpt_path).to(DEVICE)
    policy.eval()

    env = make_env()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    successes = 0
    summaries = []
    for ep in tqdm(range(args.num_episodes), desc="rollouts"):
        obs = env.reset()
        policy.reset()
        record = ep < args.num_videos
        record_frames = [render_record_frame(obs)] if record else None

        success = False
        success_steps = args.horizon
        for t in range(args.horizon):
            batch = to_policy_batch(obs)
            with torch.no_grad():
                action = policy.select_action(batch)
            action_np = action.squeeze(0).cpu().numpy()

            obs, _r, _done, _info = env.step(action_np)
            if record:
                record_frames.append(render_record_frame(obs))
            if env._check_success():
                success = True
                success_steps = t + 1
                # let it settle for a few frames so the released cube shows up in the video
                for _ in range(8):
                    obs, *_ = env.step(np.zeros(7, dtype=np.float32))
                    if record:
                        record_frames.append(render_record_frame(obs))
                break

        successes += int(success)
        summaries.append({"episode": ep, "success": success, "steps": success_steps})

        if record:
            out = VIDEO_DIR / f"eval_ep{ep:02d}_{'success' if success else 'fail'}.mp4"
            with imageio.get_writer(out, fps=20, codec="libx264", quality=8) as w:
                for f in record_frames:
                    w.append_data(f)
            print(f"  saved {out}")

    rate = successes / args.num_episodes
    summary_path = CKPT_ROOT / f"eval_summary_{ckpt_path.name}.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "num_episodes": args.num_episodes,
                "horizon": args.horizon,
                "success_rate": rate,
                "successes": successes,
                "per_episode": summaries,
            },
            indent=2,
        )
    )
    print(f"\nsuccess rate: {successes}/{args.num_episodes} = {rate:.1%}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
