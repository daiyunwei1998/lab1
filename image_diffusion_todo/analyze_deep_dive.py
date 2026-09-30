"""Follow-up experiments beyond run_analysis.py's basic FID sweep, run on demand (not part of
the main pipeline) against a single matched checkpoint per config. Three things, each solving
a specific gap identified while writing up the report:

1. Equal-budget FID (step=50000, 500 samples, all 5 configs). quad_noise/cosine_noise/
   linear_x0/linear_mean's OWN final checkpoint already IS step=50000 at 500 samples -- their
   existing final FID in fid_curve_results.json already answers this, nothing to recompute.
   Only linear_noise needs a new number here: it trained to 102000, so its existing step=50000
   entry used the cheap 100-sample curve count, not the fair 500. Comparing final-vs-final
   FIDs across configs conflates "better parameterization/schedule" with "trained twice as
   long" for linear_noise specifically -- this fixes that by holding the training budget fixed.

2. Same-seed sample grid (noise/x0/mean, step=50000). model.py's sample() draws x_T via
   torch.randn(...) and each reverse step draws z via torch.randn_like(x_t) -- same shape,
   same call count, regardless of predictor. Seeding immediately before each model's .sample()
   call reproduces the identical noise sequence across all three, so differences in the output
   are attributable to the parameterization, not to incomparable random samples.

3. Per-timestep reconstruction error (noise/x0/mean, step=50000). FID is a distributional
   statistic -- it can't say *where* a predictor fails. This asks a sharper question: at a
   fixed noise level t, how well does each predictor recover x0 from a real (not generated)
   noisy image? All three predictors' raw output gets converted to a common quantity (x_hat_0)
   using the same closed-form relations scheduler.py's step_predict_* use, so the comparison is
   apples-to-apples regardless of what each network natively predicts. Saves both the metric
   (MSE(x_hat_0, x0) per t) and the actual x_hat_0 images, since seeing *what* each predictor
   thinks the clean image looks like at a given noise level is more informative than a number.

Usage:
    python analyze_deep_dive.py [--step 50000] [--n-val-images 6]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image

from analysis_lib import compute_fid, generate_samples, load_checkpoint
from dataset import tensor_to_pil_image

DRIVE_CKPT_ROOT = "gdrive:lab1-ckpts"
DRIVE_REMOTE = "gdrive:lab1-analysis"
LOCAL_CACHE = "/tmp/ckpt_cache"
OUT_DIR = "deep_dive_out"
EVAL_DIR_CANDIDATES = ["data/afhq/eval", "data/afhq/train", "data/afhq/val"]

CONFIGS = {
    "linear_noise": {"mode": "linear", "predictor": "noise"},
    "quad_noise":   {"mode": "quad",   "predictor": "noise"},
    "cosine_noise": {"mode": "cosine", "predictor": "noise"},
    "linear_x0":    {"mode": "linear", "predictor": "x0"},
    "linear_mean":  {"mode": "linear", "predictor": "mean"},
}
PREDICTOR_LABELS = ["linear_noise", "linear_x0", "linear_mean"]
SAME_SEED_VALUE = 12345
T_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
NUM_TRAIN_TIMESTEPS = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"


def drive_ckpt_dir(label):
    cfg = CONFIGS[label]
    return f"{DRIVE_CKPT_ROOT}/{label}/predictor_{cfg['predictor']}/beta_{cfg['mode']}"


def fetch_and_load(label, step):
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    filename = f"step={step}.ckpt"
    local_path = f"{LOCAL_CACHE}/{filename}"
    result = subprocess.run(
        ["rclone", "copyto", f"{drive_ckpt_dir(label)}/{filename}", local_path,
         "--retries", "5", "--low-level-retries", "10"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {label} step={step}:\n{result.stderr}")
    return load_checkpoint(local_path, delete_after=True)


def rclone_sync(local, remote, timeout=120):
    try:
        subprocess.run(["rclone", "copy", local, remote, "--update", "-q"], timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[warn] sync {local} -> {remote} timed out, continuing", flush=True)


def find_eval_dir():
    for d in EVAL_DIR_CANDIDATES:
        if os.path.isdir(d) and glob.glob(f"{d}/*.jpg") + glob.glob(f"{d}/*.png"):
            return d
    raise RuntimeError(f"No real images found in any of {EVAL_DIR_CANDIDATES} -- run dataset.py first")


def load_val_images(n, img_size=64):
    """A handful of real images for the reconstruction-error experiment, resized/normalized
    to match training (see dataset.py) -- deterministic (sorted filenames), not random,
    so re-running this script compares against the same images every time."""
    d = find_eval_dir()
    paths = sorted(glob.glob(f"{d}/*.jpg") + glob.glob(f"{d}/*.png"))[:n]
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((img_size, img_size))
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0  # [-1, 1], matches ddpm.sample()'s output range
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs).to(device), paths


def betas_for(label, T=NUM_TRAIN_TIMESTEPS, beta_1=1e-4, beta_T=0.02):
    mode = CONFIGS[label]["mode"]
    if mode == "linear":
        return torch.linspace(beta_1, beta_T, T)
    if mode == "quad":
        return torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, T) ** 2
    if mode == "cosine":
        s = 0.008
        t = torch.arange(T + 1, dtype=torch.float64)
        f_t = torch.cos(((t / T + s) / (1 + s)) * (torch.pi / 2)) ** 2
        alpha_bar = f_t / f_t[0]
        return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()
    raise ValueError(mode)


def alpha_bar_for(label):
    return torch.cumprod(1 - betas_for(label), dim=0)


def to_x0_hat(label, x_t, t_idx, net_out):
    """Converts each predictor's native output to a common x_hat_0, using the same closed-form
    relations as scheduler.py's step_predict_* -- so the three predictors are compared on the
    same quantity regardless of what they were trained to output directly."""
    predictor = CONFIGS[label]["predictor"]
    alpha_bar = alpha_bar_for(label).to(device)
    alpha_bar_t = alpha_bar[t_idx]

    if predictor == "noise":
        x0_hat = (x_t - (1 - alpha_bar_t).sqrt() * net_out) / alpha_bar_t.sqrt()
    elif predictor == "x0":
        x0_hat = net_out
    elif predictor == "mean":
        # Invert step_predict_mean's posterior-mean formula for x0:
        #   mu~_t = A * x0 + B * x_t,  A = sqrt(abar_{t-1})*beta_t/(1-abar_t),
        #                              B = sqrt(alpha_t)*(1-abar_{t-1})/(1-abar_t)
        # => x0 = (mu~_t - B * x_t) / A
        betas = betas_for(label).to(device)
        beta_t = betas[t_idx]
        alpha_t = 1 - beta_t
        alpha_bar_t_prev = alpha_bar[t_idx - 1] if t_idx > 0 else torch.ones((), device=device)
        A = alpha_bar_t_prev.sqrt() * beta_t / (1 - alpha_bar_t)
        B = alpha_t.sqrt() * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        x0_hat = (net_out - B * x_t) / A
    else:
        raise ValueError(predictor)
    return x0_hat.clamp(-1, 1)


def equal_budget_fid(step, n_samples=500):
    print(f"=== Equal-budget FID @ step={step} ===", flush=True)
    results = {}
    for label in CONFIGS:
        ddpm = fetch_and_load(label, step)
        gen_dir = f"/tmp/gen_eqb/{label}"
        generate_samples(ddpm, n_samples, gen_dir)
        fid = compute_fid(gen_dir)
        results[label] = fid
        print(f"{label}: FID={fid:.4f}", flush=True)
        shutil.rmtree(gen_dir, ignore_errors=True)
        del ddpm
        torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"step": step, "n_samples": n_samples, "fid": results}
    json.dump(out, open(f"{OUT_DIR}/equal_budget_fid.json", "w"), indent=2)
    rclone_sync(OUT_DIR, DRIVE_REMOTE)
    return out


def same_seed_grid(step, n_samples=8):
    print(f"=== Same-seed sample grid @ step={step} ===", flush=True)
    for label in PREDICTOR_LABELS:
        predictor = CONFIGS[label]["predictor"]
        ddpm = fetch_and_load(label, step)
        torch.manual_seed(SAME_SEED_VALUE)  # same x_T + same per-step z draws across all 3 models
        target = f"{OUT_DIR}/same_seed_samples/{predictor}"
        generate_samples(ddpm, n_samples, target, batch_size=n_samples)
        print(f"{label} -> {target}", flush=True)
        del ddpm
        torch.cuda.empty_cache()
    rclone_sync(f"{OUT_DIR}/same_seed_samples", f"{DRIVE_REMOTE}/same_seed_samples")


def reconstruction_error(step, n_val_images=6):
    print(f"=== Per-timestep reconstruction error @ step={step} ===", flush=True)
    x0_batch, paths = load_val_images(n_val_images)
    print(f"Using {len(paths)} real images: {[os.path.basename(p) for p in paths]}", flush=True)

    results = {}  # {predictor: {t_frac: mse}}
    for label in PREDICTOR_LABELS:
        predictor = CONFIGS[label]["predictor"]
        ddpm = fetch_and_load(label, step)
        alpha_bar = alpha_bar_for(label).to(device)
        results[predictor] = {}

        for frac in T_FRACTIONS:
            t_idx = int(frac * (NUM_TRAIN_TIMESTEPS - 1))
            t_tensor = torch.full((x0_batch.shape[0],), t_idx, device=device, dtype=torch.long)
            eps = torch.randn_like(x0_batch)
            ab_t = alpha_bar[t_idx]
            x_t = ab_t.sqrt() * x0_batch + (1 - ab_t).sqrt() * eps

            with torch.no_grad():
                net_out = ddpm.network(x_t, timestep=t_tensor)
            x0_hat = to_x0_hat(label, x_t, t_idx, net_out)

            mse = ((x0_hat - x0_batch) ** 2).mean().item()
            results[predictor][frac] = mse
            print(f"{label}  t/T={frac}  MSE(x0_hat, x0)={mse:.4f}", flush=True)

            recon_dir = f"{OUT_DIR}/reconstruction/{predictor}/t={frac}"
            os.makedirs(recon_dir, exist_ok=True)
            for i, im in enumerate(tensor_to_pil_image(x0_hat)):
                im.save(f"{recon_dir}/{i}.png")

        del ddpm
        torch.cuda.empty_cache()

    # also save the real x0 and the noisy x_t at each t for side-by-side comparison in the report
    real_dir = f"{OUT_DIR}/reconstruction/real_x0"
    os.makedirs(real_dir, exist_ok=True)
    for i, im in enumerate(tensor_to_pil_image(x0_batch)):
        im.save(f"{real_dir}/{i}.png")

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"step": step, "t_fractions": T_FRACTIONS, "mse_by_predictor_and_t": results},
               open(f"{OUT_DIR}/reconstruction_error.json", "w"), indent=2)
    rclone_sync(OUT_DIR, DRIVE_REMOTE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=50000,
                         help="matched checkpoint step -- the training budget common to "
                              "quad_noise/cosine_noise/linear_x0/linear_mean's own final checkpoint")
    parser.add_argument("--n-val-images", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    equal_budget_fid(args.step)
    same_seed_grid(args.step)
    reconstruction_error(args.step, args.n_val_images)
    print("ALL DONE")


if __name__ == "__main__":
    main()
