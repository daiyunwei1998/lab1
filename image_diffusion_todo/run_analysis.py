"""Runs the GPU-heavy parts of the analysis (final FID table, predictor sample
images, FID-vs-training-step curve, sample progression images) on a rented pod,
and syncs the results to Google Drive as it goes.

Why this exists separately from analysis.ipynb: the same FID-vs-step sweep that
takes ~10-15s per 1000-step sample batch on a rented 5090 took ~25min on Colab's
free T4 -- a ~100x difference. Running the actual sampling/FID computation here
and syncing only the (small) resulting numbers/images to Drive means the Colab
notebook's job becomes reading already-computed data and plotting it, not
regenerating thousands of samples on a slow GPU.

Resumable, same philosophy as train_resumable.py: results are checked and
skipped if already present, and synced to Drive after every single (config,
step) FID computation -- not batched at the end -- so a pod dying mid-run
doesn't lose already-computed results (this happened once already this
session).

Usage:
    python run_analysis.py
"""
import glob
import json
import os
import re
import shutil
import subprocess

import torch
from dataset import tensor_to_pil_image
from model import DiffusionModule

device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_ROOT = "lab1-ckpts"           # local copy, pulled from Drive before running
OUT_DIR = "analysis_out"
RESULTS_JSON = f"{OUT_DIR}/fid_curve_results.json"
FINAL_JSON = f"{OUT_DIR}/final_fid_results.json"
DRIVE_REMOTE = "gdrive:lab1-analysis"

CONFIGS = {
    "linear_noise": {"mode": "linear", "predictor": "noise"},
    "quad_noise":   {"mode": "quad",   "predictor": "noise"},
    "cosine_noise": {"mode": "cosine", "predictor": "noise"},
    "linear_x0":    {"mode": "linear", "predictor": "x0"},
    "linear_mean":  {"mode": "linear", "predictor": "mean"},
}

N_SAMPLES_FINAL = 2048   # matches the assignment's own FID sample count
N_SAMPLES_CURVE = 500    # smaller, purely for wall-clock reasons across many checkpoints


def ckpt_dir(label):
    cfg = CONFIGS[label]
    return f"{CKPT_ROOT}/{label}/predictor_{cfg['predictor']}/beta_{cfg['mode']}"


def load_checkpoint(path):
    dic = torch.load(path, map_location=device, weights_only=False)
    network = dic["hparams"]["network"].to(device)
    var_scheduler = dic["hparams"]["var_scheduler"].to(device)
    predictor = dic["hparams"].get("predictor", "noise")
    ddpm = DiffusionModule(network, var_scheduler, predictor=predictor).to(device)
    ddpm.load_state_dict(dic["state_dict"])
    ddpm.eval()
    return ddpm


def generate_samples(ddpm, n, out_dir, batch_size=64):
    os.makedirs(out_dir, exist_ok=True)
    idx, remaining = 0, n
    with torch.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            samples = ddpm.sample(b)
            for im in tensor_to_pil_image(samples):
                im.save(f"{out_dir}/{idx}.png")
                idx += 1
            remaining -= b
    return out_dir


def compute_fid(gen_dir, eval_dir="data/afhq/eval"):
    result = subprocess.run(
        ["python3", "fid/measure_fid.py", eval_dir, gen_dir],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith("FID:")]
    if not lines:
        raise RuntimeError(f"FID computation failed:\n{result.stdout}\n{result.stderr}")
    return float(lines[-1].split("FID:")[1].strip())


def all_ckpt_steps(label):
    files = glob.glob(f"{ckpt_dir(label)}/step=*.ckpt")
    return sorted(int(re.search(r"step=(\d+)\.ckpt", f).group(1)) for f in files)


def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def save_and_sync(path, data):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(data, open(path, "w"), indent=2)
    subprocess.run(["rclone", "copy", OUT_DIR, DRIVE_REMOTE, "--update", "-q"])


def step_1_final_fid():
    print("=== Step 1: final FID per config ===", flush=True)
    results = load_json(FINAL_JSON)
    for label, cfg in CONFIGS.items():
        if label in results:
            print(f"[skip] {label} already done (FID={results[label]['FID']})")
            continue
        d = ckpt_dir(label)
        ts = torch.load(f"{d}/train_state.pt", map_location="cpu", weights_only=False)
        step = ts["step"]
        ddpm = load_checkpoint(f"{d}/last.ckpt")
        gen_dir = f"/tmp/gen/{label}_final"
        generate_samples(ddpm, N_SAMPLES_FINAL, gen_dir)
        fid = compute_fid(gen_dir)
        results[label] = {"scheduler": cfg["mode"], "predictor": cfg["predictor"],
                           "final_step": step, "FID": fid}
        print(f"[done] {label}: step={step} FID={fid:.4f}", flush=True)
        del ddpm
        torch.cuda.empty_cache()
        shutil.rmtree(gen_dir, ignore_errors=True)
        save_and_sync(FINAL_JSON, results)


def step_2_predictor_samples():
    print("=== Step 2: predictor sample images (8 each) ===", flush=True)
    out_dir = f"{OUT_DIR}/predictor_samples"
    for label in ["linear_noise", "linear_x0", "linear_mean"]:
        pred = CONFIGS[label]["predictor"]
        target = f"{out_dir}/{pred}"
        if os.path.exists(target) and len(glob.glob(f"{target}/*.png")) >= 8:
            print(f"[skip] {label} samples already generated")
            continue
        d = ckpt_dir(label)
        ddpm = load_checkpoint(f"{d}/last.ckpt")
        generate_samples(ddpm, 8, target, batch_size=8)
        print(f"[done] {label} -> {target}", flush=True)
        del ddpm
        torch.cuda.empty_cache()
    subprocess.run(["rclone", "copy", out_dir, f"{DRIVE_REMOTE}/predictor_samples", "--update", "-q"])


def step_3_fid_curve():
    print("=== Step 3: FID vs. training step, all configs ===", flush=True)
    results = load_json(RESULTS_JSON)
    for label in CONFIGS:
        results.setdefault(label, {})
        d = ckpt_dir(label)
        for step in all_ckpt_steps(label):
            if str(step) in results[label]:
                print(f"[skip] {label} step={step} (FID={results[label][str(step)]:.4f})")
                continue
            print(f"[run]  {label} step={step} ...", flush=True)
            ddpm = load_checkpoint(f"{d}/step={step}.ckpt")
            gen_dir = f"/tmp/gen/{label}_{step}"
            generate_samples(ddpm, N_SAMPLES_CURVE, gen_dir)
            fid = compute_fid(gen_dir)
            results[label][str(step)] = fid
            print(f"[done] {label} step={step} FID={fid:.4f}", flush=True)
            del ddpm
            torch.cuda.empty_cache()
            shutil.rmtree(gen_dir, ignore_errors=True)
            save_and_sync(RESULTS_JSON, results)  # incremental: survives a mid-run crash


def step_4_progression_images():
    print("=== Step 4: sample progression images (linear_noise) ===", flush=True)
    label = "linear_noise"
    d = ckpt_dir(label)
    steps_avail = all_ckpt_steps(label)
    chosen = steps_avail[:: max(1, len(steps_avail) // 5)][:5]
    if steps_avail[-1] not in chosen:
        chosen[-1] = steps_avail[-1]
    out_dir = f"{OUT_DIR}/progression"
    for step in chosen:
        target = f"{out_dir}/step={step}"
        if os.path.exists(target) and len(glob.glob(f"{target}/*.png")) >= 8:
            print(f"[skip] step={step} already generated")
            continue
        ddpm = load_checkpoint(f"{d}/step={step}.ckpt")
        generate_samples(ddpm, 8, target, batch_size=8)
        print(f"[done] step={step} -> {target}", flush=True)
        del ddpm
        torch.cuda.empty_cache()
    subprocess.run(["rclone", "copy", out_dir, f"{DRIVE_REMOTE}/progression", "--update", "-q"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    step_1_final_fid()
    step_2_predictor_samples()
    step_4_progression_images()
    step_3_fid_curve()  # last: by far the most expensive step
    print("ALL DONE")


if __name__ == "__main__":
    main()
