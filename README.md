# ACT on PickPlaceCube (Robosuite)

End-to-end Action Chunking Transformer (ACT) pipeline for a Franka pick-and-place task in
Robosuite/MuJoCo: build a custom env → collect demos with a scripted controller →
train ACT via LeRobot → evaluate in sim.

A red cube spawns on the left half of the table, a green visual target spawns on the
right half — **both randomized every episode**. The trained ACT policy receives only
camera images + proprioception (no privileged target position).

**Headline result:** **48 / 50 = 96 %** success on random cube + random target,
80.79 ± 2.87 sim steps to success.

---

## 1 · Prerequisites

* Linux x86-64 with an NVIDIA GPU (≥ 12 GB VRAM recommended; tested on RTX 3090 24 GB)
* CUDA-capable driver compatible with PyTorch 2.11 / CUDA 13 wheels (`nvidia-smi` should work)
* `uv` (the package/venv manager) — install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
* Headless OpenGL via EGL (Robosuite/MuJoCo use it for offscreen rendering).
  This is set automatically as `MUJOCO_GL=egl` inside every script — no system
  package install required on most Ubuntu/Debian boxes.
* `git` (the project pulls Robosuite v1.5.1 from source)

> No system FFmpeg needed. The dataset is image-backed (PNG frames) and `imageio[ffmpeg]`
> bundles the binary required for writing eval mp4s.

---

## 2 · Install

```bash
git clone <this-repo>            # or copy the project folder
cd act

# Create the venv and resolve all deps from pyproject.toml + uv.lock.
uv sync
```

`uv sync` will:
* Provision Python 3.10 (`.python-version`)
* Install PyTorch (CUDA build), Robosuite v1.5.1 from git, LeRobot 0.3.2,
  Robomimic, and supporting libs (h5py, imageio, av, tensorboard, matplotlib, …)
* Pre-build a stub `egl_probe` if it complains (already vendored in the venv if needed)

**Verify:**
```bash
uv run python - <<'PY'
import torch, lerobot, robosuite
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("robosuite", robosuite.__version__, "lerobot", lerobot.__version__)
PY
```

Expected:
```
torch 2.11.0+cu130 cuda True
robosuite 1.5.1 lerobot 0.3.2
```

---

## 3 · Pipeline

Four scripts run end-to-end. Pass `--help` to any of them for the full CLI.

### 3a · Verify the simulator + scripted policy

```bash
uv run python scripts/06_pickplace_demo_check.py
```

Renders one full scripted-policy episode at 256×256 and writes
`videos/verification/pickplace_scripted.mp4`. Use this to confirm
the env loads, the cube + green target are visible, and the
phase-machine controller succeeds.

### 3b · Collect 100 demos

```bash
uv run python scripts/07_pickplace_collect.py --num-demos 100
```

Streams scripted-policy rollouts (cube + target both randomized every reset)
straight into a LeRobot v2.1 dataset at `data/lerobot/pick_place_cube/`.
Failed scripted attempts are skipped — you'll typically see 100 / 100.

Output: ~240 MB, 100 episodes, ~8 800 frames, two camera streams (`agentview`,
wrist) at 84×84, 9-d proprio, 7-d OSC action, fps 20. Takes ~3 minutes.

CLI:
```
--num-demos N    target number of successful episodes (default 100)
--seed S         starting seed (default 0)
```

### 3c · Train ACT

```bash
uv run python scripts/08_pickplace_train.py
```

Trains for 30 000 steps with batch 64, LR 1e-5, ResNet18 backbone (ImageNet
pretrained), CVAE encoder, action chunk size 100. Writes:

* `checkpoints/act_pickplace/step_{05000…30000}/` — every 5 k steps
* `checkpoints/act_pickplace/last/` — final policy (safetensors + config)
* `logs/tb/act_pickplace/` — TensorBoard scalars (loss, l1, kld, grad, sps)

~63 minutes on an RTX 3090 (~12 GB VRAM, 8 sps). To change steps / batch /
chunk size etc., edit the constants at the top of `scripts/08_pickplace_train.py`.

Watch live:
```bash
uv run tensorboard --logdir logs/tb
```

### 3d · Evaluate in sim

```bash
uv run python scripts/09_pickplace_eval.py
```

Loads `checkpoints/act_pickplace/last/` and rolls it out for 50 episodes in
`PickPlaceCube`. Saves:

* `videos/eval_pickplace/eval_ep0[0-5]_{success,fail}.mp4` — first 6 rollouts
  rendered from a 256×256 frontview
* `checkpoints/act_pickplace/eval_summary_last.json` — full per-episode log
  (success bool, steps to success)

CLI:
```
--ckpt NAME            checkpoint dir under checkpoints/act_pickplace/ (default: last)
--num-episodes N       (default 50)
--num-videos N         how many of the first N rollouts to record (default 6)
--seed S               eval seed (default 1000, disjoint from collection)
--horizon T            max sim steps per episode (default 250)
```

Want to test closed-loop ACT (the residual-failure hypothesis from the report)?
Edit `scripts/09_pickplace_eval.py` and just below `policy.eval()` add e.g.
`policy.config.n_action_steps = 16` — no retrain needed, the chunk size 100 model
will simply re-query the camera every 16 actions instead of every 100.

### 3e · Regenerate the loss plot

```bash
uv run python scripts/make_loss_plot.py
```

Reads `logs/tb/act_pickplace/` and writes `report/training_loss.png`.

---

## 4 · Layout

```
.
├── pyproject.toml / uv.lock       # uv project (Python 3.10)
├── README.md                       # this file
├── report.md                       # 1–2 page write-up (assignment deliverable)
├── report/
│   ├── training_loss.png            # loss vs step (linear + log)
│   └── eval_5successes.mp4          # 5 concatenated successful eval rollouts
│
├── scripts/
│   ├── envs/
│   │   ├── pick_place_cube.py        # custom env (extends Robosuite Lift)
│   │   └── scripted_policy.py        # phase-machine OSC P-controller
│   ├── 06_pickplace_demo_check.py    # smoke-test the scripted policy
│   ├── 07_pickplace_collect.py       # scripted demos → LeRobot dataset
│   ├── 08_pickplace_train.py         # train ACT, write TensorBoard
│   ├── 09_pickplace_eval.py          # 50-episode rollout
│   └── make_loss_plot.py             # TB scalars → loss PNG
│
├── data/lerobot/pick_place_cube/   # LeRobot v2.1 dataset (image-backed, 84×84, 2 cams)
├── videos/
│   ├── verification/                 # env snapshots + scripted-policy mp4
│   └── eval_pickplace/               # individual ACT eval rollouts
├── checkpoints/act_pickplace/      # ACT weights + JSON eval summary
└── logs/tb/act_pickplace/          # TensorBoard event files
```

---

## 5 · Env / data shape

| Field | Value |
|---|---|
| Robot | Franka Panda + `OSC_POSE` controller, 7-d action `[dx,dy,dz,droll,dpitch,dyaw,gripper]` |
| State (proprio) | `eef_pos (3) + eef_quat (4) + gripper_qpos (2) = 9` |
| Cameras | `agentview` + `robot0_eye_in_hand` (both 84×84 RGB) |
| Control / dataset FPS | 20 Hz |
| Cube spawn | uniform in a 4 cm × 8 cm box on the left half of the table |
| Target spawn | uniform in a 10 cm × 16 cm box on the right half (visible green square, no proprio leakage) |
| Success criterion | `‖cube_xy − target_xy‖ < 4.5 cm` AND cube near table height AND gripper released |
| Demos | 100 scripted (100 % yield) |
| Avg episode length | ~88 frames |

---

## 6 · Reproducing the headline numbers

```bash
uv sync
uv run python scripts/07_pickplace_collect.py --num-demos 100
uv run python scripts/08_pickplace_train.py
uv run python scripts/09_pickplace_eval.py
uv run python scripts/make_loss_plot.py
```

End-to-end, this takes ~70 minutes on a single RTX 3090.

The trained checkpoint and eval JSON in this repo are the run reported in
`report.md` (96 % over 50 random-target episodes). Re-runs may differ within a
small noise band because of dataloader shuffling and CUDA non-determinism.

---

## 7 · Troubleshooting

* **`OSError: libavutil.so.57: cannot open shared object file` when loading the dataset** —
  the dataset is image-backed (`use_videos=False`); you should not hit this. If
  you do, you almost certainly switched a feature to `dtype: "video"`. Either
  install system FFmpeg 5/6 or revert to image dtype.
* **`ModuleNotFoundError: No module named 'egl_probe'`** — `uv sync` should not
  reinstall this from PyPI (the source build needs CMake + dev libs). The repo
  ships a small stub at `.venv/lib/python3.10/site-packages/egl_probe/` that
  satisfies Robomimic's only call site (returning a default GPU index). If your
  venv is missing it, recreate it manually (5-line file in `report.md` §1
  history, or just `mkdir` the folder and add an `__init__.py` returning `[0]`
  from `get_available_devices()`).
* **MuJoCo segfault / black frames** — make sure `MUJOCO_GL=egl` is set
  (the scripts do this automatically) and that the GPU has a working driver.
  Run `nvidia-smi` to confirm.
* **Training stuck at step 1 / 0.5 sps** — the first batch always pays for
  dataloader + ResNet18 weight download. After that you should see ≥7 sps.
* **`PYTHONPATH` errors when importing `scripts.envs`** — every entry-point
  script does `sys.path.insert(0, project_root)` itself, but if you're calling
  Python by hand from another directory, prepend `PYTHONPATH=$PWD`.
