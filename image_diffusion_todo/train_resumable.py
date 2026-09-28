"""Mirror of train.py with checkpoint-resume added, for unattended rented-GPU
runs where the pod can die mid-training.

train.py itself is untouched -- this is a separate copy of its training loop,
not a patch to it, so the teacher's original script stays exactly as given.

Why a plain re-run of train.py can't survive an interruption:
  ddpm.save()/load() (model.py, also untouched) only persist the model's
  weights -- not the step count, optimizer momentum, LR-warmup state, or
  loss history. So even with last.ckpt intact, `python train.py` always
  starts a fresh DiffusionModule at step 0; there's no way to continue
  training from where it left off using only what train.py already saves.

What this script adds:
  A second file, train_state.pt, saved alongside last.ckpt at the same
  cadence, holding step/optimizer/lr_scheduler/losses. On startup, if both
  last.ckpt and train_state.pt already exist, this script loads both and
  continues the loop from the saved step instead of 0.

  save_dir is fixed per (mode, predictor) -- no timestamp subfolder like
  train.py uses -- specifically so that re-running the exact same command
  after a crash finds its own previous output automatically, with no
  manual path-hunting needed at 3am when a pod just died.
  Consequence: to intentionally start a config over from scratch, delete
  its results/predictor_{predictor}/beta_{mode}/ directory first, or this
  script will resume the old run instead.

Usage: identical CLI to train.py, e.g.
    python train_resumable.py --mode linear --predictor noise --train_num_steps 100000
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import (AFHQDataModule, get_data_iterator, save_traj_strip,
                     tensor_to_pil_image)
from dotmap import DotMap
from model import DiffusionModule
from network import UNet
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from tqdm import tqdm
from PIL import Image

matplotlib.use("Agg")


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    # Fixed, non-timestamped save_dir (unlike train.py) so a re-run after a
    # crash lands in the same place and can find its own last.ckpt/train_state.pt.
    if args.use_cfg:
        save_dir = Path(f"results/cfg_predictor_{args.predictor}/beta_{config.mode}")
    else:
        save_dir = Path(f"results/predictor_{args.predictor}/beta_{config.mode}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    image_resolution = 64
    ds_module = AFHQDataModule(
        "./data",
        batch_size=config.batch_size,
        num_workers=4,
        max_num_images_per_cat=config.max_num_images_per_cat,
        image_resolution=image_resolution
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode=config.mode,
    )

    network = UNet(
        T=config.num_diffusion_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    ddpm = DiffusionModule(network, var_scheduler, predictor=config.predictor)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    ckpt_path = save_dir / "last.ckpt"
    state_path = save_dir / "train_state.pt"

    step = 0
    losses = []
    images = []

    if ckpt_path.exists() and state_path.exists():
        print(f"Found existing checkpoint + train state in {save_dir} -- resuming.")
        ddpm.load(str(ckpt_path))
        ddpm = ddpm.to(config.device)
        train_state = torch.load(state_path, map_location=config.device)
        optimizer.load_state_dict(train_state["optimizer"])
        scheduler.load_state_dict(train_state["scheduler"])
        step = train_state["step"]
        losses = train_state["losses"]
        print(f"Resumed at step {step}/{config.train_num_steps} "
              f"({len(losses)} logged losses).")
    elif ckpt_path.exists() or state_path.exists():
        print(f"WARNING: only one of last.ckpt / train_state.pt exists in {save_dir} "
              f"(previous run likely died mid-save). Starting fresh from step 0 "
              f"rather than risk loading a half-written pair.")

    def save_everything():
        ddpm.save(str(ckpt_path))
        torch.save({
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "losses": losses,
        }, state_path)

    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                ddpm.eval()

                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                samples = ddpm.sample(4, return_traj=False)
                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                traj = ddpm.sample(1, return_traj=True)
                save_traj_strip(save_dir / f"step={step}-traj.png", traj, num_frames=10, pad=4)

                save_everything()
                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device), label.to(config.device)

            if args.use_cfg:
                loss = ddpm.get_loss(img, class_label=label)
            else:
                loss = ddpm.get_loss(img)

            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)

    plt.plot(losses)
    plt.savefig(f"{save_dir}/loss.png")
    plt.close()
    save_everything()
    print(f"Saved the final checkpoint at step {step} to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=3000,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=64)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--predictor", type=str, default="noise",
                    choices=["noise", "x0", "mean"],
                    help="which parameterization the model uses")
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "cosine", "quad"],
                        help="beta scheduling mode")

    args = parser.parse_args()
    main(args)
