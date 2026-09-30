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

Real-time Drive sync (optional, off by default):
  Pass --drive_remote gdrive:lab1-ckpts/<label> to sync results/ to Google Drive
  right after every checkpoint write (not on a separate timer -- tied directly to
  the moment new data actually exists, so it's as close to real-time as syncing
  can be without wastefully re-uploading unchanged data in between).
  Prints "SYNC OK at step N" after each one, and a distinct
  "FINAL SYNC COMPLETE -- SAFE TO DELETE INSTANCE" line after the last one, once
  training has fully finished and that last sync has been confirmed to succeed --
  that line is the actual signal it's safe to tear the pod down, not just the
  "Saved the final checkpoint" line (which only means training is done, not that
  the data has left the pod yet).

Real-time FID (optional, off by default):
  Pass --analysis_remote gdrive:lab1-analysis (and --fid_interval, default 10000)
  to compute FID right after each periodic/final checkpoint save, using the model
  already resident in memory -- no round trip of uploading a checkpoint to Drive
  and later re-downloading it in a separate analysis pass just to sample from it.
  Results merge into the SAME fid_curve_results.json / error_analysis_{label}.json
  / worst_samples/ layout run_analysis.py produces, so analysis.ipynb doesn't care
  which of the two computed a given number. Since another pod's run_analysis.py
  may be writing to the same fid_curve_results.json concurrently, updates fetch
  the current remote copy first and merge into it rather than overwriting --
  hit a Drive sync conflict earlier this session from two writers doing a naive
  overwrite, this avoids repeating that.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from analysis_lib import (compute_fid, compute_fid_with_error_analysis,
                          generate_samples)
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

ANALYSIS_OUT = "analysis_out"
N_SAMPLES_FINAL = 500
N_SAMPLES_CURVE = 100


def fetch_remote_results(analysis_remote, timeout=60):
    """Best-effort read of the current remote fid_curve_results.json, so a merge-then-write
    doesn't clobber another pod's entries for other configs/steps."""
    if not analysis_remote:
        return {}
    tmp = f"{ANALYSIS_OUT}/.remote_fetch_tmp.json"
    try:
        r = subprocess.run(
            ["rclone", "copyto", f"{analysis_remote}/fid_curve_results.json", tmp],
            capture_output=True, timeout=timeout,
        )
        if r.returncode == 0 and os.path.exists(tmp):
            data = json.load(open(tmp))
            os.remove(tmp)
            return data
    except Exception:
        pass
    return {}


def run_analysis_at_step(ddpm, step, label, analysis_remote, is_final):
    """Computes one FID curve point (or, at the final checkpoint, the full 500-sample
    headline FID + error analysis) using ddpm as it currently sits in memory -- the whole
    point being that this needs no checkpoint download, since training already has it."""
    was_training = ddpm.training
    ddpm.eval()
    try:
        if is_final:
            gen_dir = f"{ANALYSIS_OUT}/final_samples/{label}"
            generate_samples(ddpm, N_SAMPLES_FINAL, gen_dir)
            fid = compute_fid_with_error_analysis(gen_dir, ANALYSIS_OUT, label)
        else:
            gen_dir = f"/tmp/gen_{label}_{step}"
            generate_samples(ddpm, N_SAMPLES_CURVE, gen_dir)
            fid = compute_fid(gen_dir)
            shutil.rmtree(gen_dir, ignore_errors=True)
    finally:
        if was_training:
            ddpm.train()

    os.makedirs(ANALYSIS_OUT, exist_ok=True)
    results = fetch_remote_results(analysis_remote)
    results.setdefault(label, {})[str(step)] = fid
    results_path = f"{ANALYSIS_OUT}/fid_curve_results.json"
    json.dump(results, open(results_path, "w"), indent=2)
    print(f"[analysis] {label} step={step} FID={fid:.4f}", flush=True)

    if analysis_remote:
        subprocess.Popen(
            ["rclone", "copy", ANALYSIS_OUT, analysis_remote, "--update", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return fid


def sync_to_drive(drive_remote: str, step: int, final: bool = False) -> None:
    if not drive_remote:
        return
    cmd = ["rclone", "copy", "results", drive_remote, "--update", "-q",
           "--low-level-retries", "3", "--retries", "1"]
    if not final:
        # Periodic syncs are fire-and-forget: a Drive-side hiccup (rate limits on
        # rclone's shared client_id, a stalled connection -- no timeout on rclone's
        # own retry loop otherwise) must never block the training loop. We accept
        # not knowing this particular sync's outcome in exchange for training never
        # freezing on it; the NEXT periodic sync (or the final one) picks up
        # whatever this one missed, since --update only copies what's actually new.
        subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"SYNC started (background, not awaited) at step {step}")
        return
    # The final sync must actually be confirmed before declaring it safe to
    # delete the instance, so this one blocks -- but with a hard timeout, so a
    # hung connection can't freeze the script forever with no signal either way.
    try:
        result = subprocess.run(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"SYNC FAILED at step {step} (timed out after 600s) -- "
              f"NOT safe to delete the instance yet.")
        return
    if result.returncode != 0:
        print(f"SYNC FAILED at step {step} (rclone exit {result.returncode}) -- "
              f"NOT safe to delete the instance yet.")
        return
    print("FINAL SYNC COMPLETE -- SAFE TO DELETE INSTANCE")


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

    # NOT named `label` -- the training loop below reuses that name for the
    # per-batch AFHQ class-label tensor from next(train_it), which would shadow it.
    config_label = f"{config.mode}_{config.predictor}"

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
                # Permanent, step-stamped snapshot of the model weights -- unlike
                # last.ckpt (overwritten every log_interval), this one is never
                # replaced, so any earlier step's model can still be reloaded and
                # sampled from later. No disk-space gating: rented-pod disks aren't
                # the constrained local disk this used to be written for.
                ddpm.save(str(save_dir / f"step={step}.ckpt"))
                sync_to_drive(args.drive_remote, step)

                if args.analysis_remote and step > 0 and step % args.fid_interval == 0:
                    run_analysis_at_step(ddpm, step, config_label, args.analysis_remote, is_final=False)

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
    ddpm.save(str(save_dir / f"step={step}.ckpt"))
    print(f"Saved the final checkpoint at step {step} to {ckpt_path}")

    if args.analysis_remote:
        run_analysis_at_step(ddpm, step, config_label, args.analysis_remote, is_final=True)

    sync_to_drive(args.drive_remote, step, final=True)


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
        "--drive_remote", type=str, default="",
        help="rclone remote:path (e.g. gdrive:lab1-ckpts/linear_noise) to sync "
             "results/ to after every checkpoint write. Empty = sync disabled.",
    )
    parser.add_argument(
        "--analysis_remote", type=str, default="",
        help="rclone remote:path (e.g. gdrive:lab1-analysis) to compute and sync FID to "
             "right after each periodic/final checkpoint, using the model already in "
             "memory -- no separate analysis pass needed. Empty = disabled.",
    )
    parser.add_argument(
        "--fid_interval", type=int, default=10000,
        help="compute a FID curve point every N steps when --analysis_remote is set "
             "(independent of --log_interval, since FID is far more expensive than the "
             "loss-plot/sample-preview saves log_interval triggers).",
    )
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
