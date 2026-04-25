"""Train ACT on the PickPlaceCube dataset.

Same loop as scripts/04_train_act.py — just different dataset / output paths.
TB:  uv run tensorboard --logdir logs/tb
"""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "lerobot" / "pick_place_cube"
CKPT_DIR = ROOT / "checkpoints" / "act_pickplace"
TB_DIR = ROOT / "logs" / "tb" / "act_pickplace"

REPO_ID = "local/pick_place_cube"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
NUM_WORKERS = 4
TOTAL_STEPS = 30_000
LOG_EVERY = 100
SAVE_EVERY = 5_000
LR = 1e-5
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4

CHUNK_SIZE = 100
N_ACTION_STEPS = 100


def build_policy(dataset: LeRobotDataset) -> ACTPolicy:
    state_dim = dataset.features["observation.state"]["shape"][0]
    action_dim = dataset.features["action"]["shape"][0]
    image_keys = [k for k in dataset.features if k.startswith("observation.images.")]

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    for k in image_keys:
        h, w, c = dataset.features[k]["shape"]
        input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        device=DEVICE,
        push_to_hub=False,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        optimizer_lr=LR,
        optimizer_lr_backbone=LR_BACKBONE,
        optimizer_weight_decay=WEIGHT_DECAY,
    )
    return ACTPolicy(cfg, dataset_stats=dataset.meta.stats)


def make_optimizer(policy: ACTPolicy) -> torch.optim.Optimizer:
    cfg = policy.config
    backbone_params, other_params = [], []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.optimizer_lr},
            {"params": backbone_params, "lr": cfg.optimizer_lr_backbone},
        ],
        weight_decay=cfg.optimizer_weight_decay,
    )


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    delta_timestamps = {"action": [i / 20.0 for i in range(CHUNK_SIZE)]}
    dataset = LeRobotDataset(REPO_ID, root=DATA_ROOT, delta_timestamps=delta_timestamps)
    print(f"dataset: {dataset.num_episodes} eps / {dataset.num_frames} frames @ {dataset.fps} fps")

    policy = build_policy(dataset).to(DEVICE)
    policy.train()
    n_param = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params: {n_param / 1e6:.2f} M")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    optim = make_optimizer(policy)
    writer = SummaryWriter(TB_DIR)

    (CKPT_DIR / "train_config.json").write_text(
        json.dumps(
            {
                "batch_size": BATCH_SIZE,
                "total_steps": TOTAL_STEPS,
                "lr": LR,
                "lr_backbone": LR_BACKBONE,
                "weight_decay": WEIGHT_DECAY,
                "chunk_size": CHUNK_SIZE,
                "n_action_steps": N_ACTION_STEPS,
                "device": DEVICE,
            },
            indent=2,
        )
    )

    step = 0
    t0 = time.perf_counter()
    loader_iter = iter(loader)
    while step < TOTAL_STEPS:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {
            k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        loss, info = policy.forward(batch)

        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optim.step()

        step += 1
        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("train/steps_per_sec", sps, step)
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"train/{k}", v, step)
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    writer.add_scalar(f"train/{k}", v.item(), step)
            print(
                f"step {step:>6}/{TOTAL_STEPS}  loss {loss.item():.4f}  "
                f"grad {float(grad_norm):.3f}  {sps:.1f} steps/s",
                flush=True,
            )

        if step % SAVE_EVERY == 0 or step == TOTAL_STEPS:
            ckpt = CKPT_DIR / f"step_{step:07d}"
            policy.save_pretrained(ckpt)
            print(f"saved checkpoint -> {ckpt}", flush=True)

    last = CKPT_DIR / "last"
    if last.exists():
        import shutil
        shutil.rmtree(last)
    policy.save_pretrained(last)
    print(f"final policy -> {last}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
