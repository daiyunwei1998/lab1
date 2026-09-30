"""Computes FID for one already-saved local checkpoint, standalone -- meant to be launched
as a background subprocess by train_resumable.py right after a checkpoint write, so the
training loop's gradient steps can continue immediately instead of blocking for the several
minutes a 100/500-sample FID computation takes.

Runs as a separate process (not a thread) specifically so it can hold its own CUDA context
and model copy independent of the training process's -- the training process keeps optimizing
the SAME live model object, so anything computing FID from it would need to either block
training or risk sampling from a model that's still being mutated mid-computation. Reading a
finished, already-saved checkpoint file sidesteps that entirely.

train_resumable.py bounds how many of these can be in flight at once (waits for the previous
one to finish before launching the next) rather than firing them off unbounded -- a slow
analysis run or Drive sync then only delays the *next* checkpoint's analysis launch, never
corrupts or races with the training loop itself.

Usage:
    python analyze_checkpoint.py --ckpt_path <path> --label <label> --step <N> \
        [--is_final] [--analysis_remote gdrive:lab1-analysis]
"""
import argparse
import shutil
import subprocess

from analysis_lib import (ANALYSIS_OUT_DEFAULT, compute_fid, compute_fid_with_error_analysis,
                          generate_samples, load_checkpoint, merge_fid_result)

N_SAMPLES_FINAL = 500
N_SAMPLES_CURVE = 100


def main(args):
    ddpm = load_checkpoint(args.ckpt_path, delete_after=False)

    if args.is_final:
        gen_dir = f"{args.out_dir}/final_samples/{args.label}"
        generate_samples(ddpm, N_SAMPLES_FINAL, gen_dir)
        fid = compute_fid_with_error_analysis(gen_dir, args.out_dir, args.label)
    else:
        gen_dir = f"/tmp/gen_{args.label}_{args.step}"
        generate_samples(ddpm, N_SAMPLES_CURVE, gen_dir)
        fid = compute_fid(gen_dir)
        shutil.rmtree(gen_dir, ignore_errors=True)

    merge_fid_result(args.out_dir, args.analysis_remote, args.label, args.step, fid)
    print(f"[analysis] {args.label} step={args.step} FID={fid:.4f}", flush=True)

    if args.analysis_remote:
        subprocess.run(
            ["rclone", "copy", args.out_dir, args.analysis_remote, "--update", "-q"],
            timeout=90,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--is_final", action="store_true")
    parser.add_argument("--analysis_remote", default="")
    parser.add_argument("--out_dir", default=ANALYSIS_OUT_DEFAULT)
    args = parser.parse_args()
    main(args)
