"""Read TensorBoard scalars and write a training-loss curve PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
TB_DIR = ROOT / "logs" / "tb" / "act_pickplace"
OUT = ROOT / "report" / "training_loss.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ea = EventAccumulator(str(TB_DIR))
    ea.Reload()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tag, label, color in [
        ("train/loss", "total loss", "C0"),
        ("train/l1_loss", "L1 (action recon)", "C1"),
        ("train/kld_loss", "KLD (CVAE)", "C2"),
    ]:
        ev = ea.Scalars(tag)
        steps = [e.step for e in ev]
        vals = [e.value for e in ev]
        ax[0].plot(steps, vals, label=label, color=color, linewidth=1.2)

    ax[0].set_xlabel("training step")
    ax[0].set_ylabel("loss")
    ax[0].set_title("ACT training losses (linear)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    for tag, label, color in [
        ("train/loss", "total loss", "C0"),
        ("train/l1_loss", "L1 (action recon)", "C1"),
    ]:
        ev = ea.Scalars(tag)
        steps = [e.step for e in ev]
        vals = [e.value for e in ev]
        ax[1].plot(steps, vals, label=label, color=color, linewidth=1.2)

    ax[1].set_yscale("log")
    ax[1].set_xlabel("training step")
    ax[1].set_ylabel("loss (log)")
    ax[1].set_title("ACT training losses (log)")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend()

    fig.suptitle("ACT — PickPlaceCube — 30k steps, batch 64", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
