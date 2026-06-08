#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.pure_motion.imu_utils import DEFAULT_SENSOR_COMBOS, build_f_imu, build_f_imu_selected
from scripts.eval.eval_imu_streaming import (
    load_pose_evaluator,
    render_side_by_side,
    stream_predict_sequence,
    summarize_errors,
)


IMUPOSER_DEVICE_TO_SLOT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}

METRIC_NAMES = [
    "sip_deg",
    "angular_deg",
    "masked_angular_deg",
    "pos_cm",
    "masked_pos_cm",
    "mesh_cm",
    "jitter_100mps3",
    "distance_cm",
]

DEFAULT_SELECTION = "angular_deg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Best-of-K online IMU evaluation on IMUPoser sequences"
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/root/autodl-tmp/dataset/processed/eval/imuposer_test.pt",
    )
    parser.add_argument("--combo", type=str, default="lw_rp_h")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--ddim", type=int, default=None, help="Override gen-only DDIM steps.")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="Diffusion sampler used during inference.",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Override online chunk size.")
    parser.add_argument("--history-frames", type=int, default=None, help="Override online history length.")
    parser.add_argument(
        "--known-gt-frames",
        type=int,
        default=0,
        help="Use the first N GT motion frames as observed warm-start history for online inference.",
    )
    parser.add_argument(
        "--inject-observed-prefix-into-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the observed GT prefix directly into predicted output before autoregressive generation.",
    )
    parser.add_argument(
        "--init-rollout-from-observed-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize rollout state from the observed GT prefix.",
    )
    parser.add_argument(
        "--skip-first-frames",
        type=int,
        default=0,
        help="Ignore the first N frames during metric evaluation. Comparison starts from frame N+1.",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--selection",
        type=str,
        default=DEFAULT_SELECTION,
        choices=METRIC_NAMES + ["composite"],
        help="Criterion used to choose one best sample per sequence for saving and visualization.",
    )
    parser.add_argument(
        "--all-seqs",
        action="store_true",
        help="Run all IMUPoser test sequences. Default is only the first sequence.",
    )
    parser.add_argument(
        "--render-best-count",
        type=int,
        default=0,
        help="Render best sample for the first N evaluated sequences.",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        default="aitviewer",
        choices=["aitviewer", "opencv"],
    )
    return parser.parse_args()


def load_model_with_overrides(
    ckpt_path,
    exp_name,
    ddim_steps=None,
    chunk_size=None,
    history_frames=None,
    sampler="ddim",
):
    config_dir = Path("/home/project/GENMO/configs")
    overrides = [f"exp={exp_name}", "use_wandb=false"]
    if ddim_steps is not None:
        overrides.append(f"+model_cfg.diffusion.gen_only_test_timestep_respacing={ddim_steps}")
    if sampler is not None:
        overrides.append(f"model_cfg.diffusion.sampler={sampler}")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    model = instantiate(cfg.model, _recursive_=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.pipeline.args.use_cfg_sampler_for_gen = False
    model.pipeline.denoiser3d.args.use_cfg_sampler_for_gen = False
    if chunk_size is not None:
        model.streaming_online_chunk_frames = int(chunk_size)
    if history_frames is not None:
        model.streaming_history_frames = int(history_frames)
    model = model.cuda().eval()
    return model


def build_model_input(acc5, ori5, model, combo_name):
    length = acc5.shape[0]
    acc7 = torch.zeros(length, 7, 3, dtype=torch.float32)
    ori7 = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(length, 7, 1, 1)
    for src_idx, dst_idx in IMUPOSER_DEVICE_TO_SLOT.items():
        acc7[:, dst_idx] = acc5[:, src_idx]
        ori7[:, dst_idx] = ori5[:, src_idx]

    sensor_ids = DEFAULT_SENSOR_COMBOS[combo_name]
    imu_dim = int(model.model_cfg.imu_dim)
    if imu_dim == 70:
        sensor_mask = torch.zeros(7, dtype=torch.float32)
        sensor_mask[sensor_ids] = 1.0
        for slot_idx in range(7):
            if sensor_mask[slot_idx] == 0:
                acc7[:, slot_idx] = 0
                ori7[:, slot_idx] = 0
        f_imu, _ = build_f_imu(acc7, ori7, sensor_mask, include_combo_mask=True)
        return f_imu, sensor_mask

    if imu_dim == 36:
        f_imu, _, _, _ = build_f_imu_selected(acc7, ori7, sensor_ids, rotation_rep="mat9")
        sensor_mask = torch.ones(len(sensor_ids), dtype=torch.float32)
        return f_imu, sensor_mask

    raise ValueError(f"Unsupported imu_dim: {imu_dim}")


def format_metric_dict(metrics):
    return {k: round(v, 4) for k, v in metrics.items()}


def format_live_seed_message(seq_idx, seed, elapsed_s, metrics_mean):
    return (
        f"CURRENT_SEED seq={seq_idx + 1} "
        f"seed={seed} "
        f"time={elapsed_s:.2f}s "
        f"angular={metrics_mean['angular_deg']:.4f} "
        f"mesh={metrics_mean['mesh_cm']:.4f}"
    )


def compute_composite_scores(samples):
    opt_keys = [
        "sip_deg",
        "angular_deg",
        "masked_angular_deg",
        "pos_cm",
        "masked_pos_cm",
        "mesh_cm",
        "distance_cm",
    ]
    means = {k: sum(s["metrics_mean"][k] for s in samples) / len(samples) for k in opt_keys}
    stds = {
        k: (sum((s["metrics_mean"][k] - means[k]) ** 2 for s in samples) / len(samples)) ** 0.5 + 1e-8
        for k in opt_keys
    }
    for sample in samples:
        sample["composite_z"] = sum(
            (sample["metrics_mean"][k] - means[k]) / stds[k] for k in opt_keys
        )


def pick_best_sample(samples, selection):
    if selection == "composite":
        compute_composite_scores(samples)
        return min(samples, key=lambda x: x["composite_z"])
    return min(samples, key=lambda x: x["metrics_mean"][selection])


def save_summary_text(path, summary_tensor):
    with open(path, "w") as handle:
        for idx, name in enumerate(METRIC_NAMES):
            mean = summary_tensor[idx, 0].item()
            std = summary_tensor[idx, 1].item()
            handle.write(f"{name}: {mean:.4f} (+/- {std:.4f})\n")


def evaluate_sequence(
    model,
    evaluator,
    seq_idx,
    acc5,
    ori5,
    pose_t,
    tran_t,
    combo,
    window,
    k,
    seed_start,
    known_gt_frames=0,
    skip_first_frames=0,
    inject_observed_prefix_into_output=False,
    init_rollout_from_observed_prefix=False,
):
    f_imu, sensor_mask = build_model_input(acc5, ori5, model, combo)
    observed_pose_prefix = None
    observed_tran_prefix = None
    if known_gt_frames > 0:
        observed_pose_prefix = pose_t[:known_gt_frames].clone()
        observed_tran_prefix = tran_t[:known_gt_frames].clone()
    samples = []
    for offset in range(k):
        seed = seed_start + offset
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pose_p, tran_p = stream_predict_sequence(
            model,
            f_imu,
            sensor_mask,
            window,
            observed_pose_prefix=observed_pose_prefix,
            observed_tran_prefix=observed_tran_prefix,
            inject_observed_prefix_into_output=inject_observed_prefix_into_output,
            init_rollout_from_observed_prefix=init_rollout_from_observed_prefix,
            show_progress=True,
            progress_desc=f"seq {seq_idx + 1} seed {seed}",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if skip_first_frames > 0:
            pose_eval_p = pose_p[skip_first_frames:]
            pose_eval_t = pose_t[skip_first_frames:]
            tran_eval_p = tran_p[skip_first_frames:]
            tran_eval_t = tran_t[skip_first_frames:]
        else:
            pose_eval_p = pose_p
            pose_eval_t = pose_t
            tran_eval_p = tran_p
            tran_eval_t = tran_t
        if pose_eval_p.shape[0] == 0:
            raise ValueError(
                f"skip_first_frames={skip_first_frames} removes the whole sequence "
                f"(sequence_index={seq_idx}, total_frames={pose_p.shape[0]})."
            )
        err = evaluator.eval(
            pose_eval_p.cuda(),
            pose_eval_t.cuda(),
            tran_p=tran_eval_p.cuda(),
            tran_t=tran_eval_t.cuda(),
        ).cpu()
        metrics_mean = {name: float(err[i, 0].item()) for i, name in enumerate(METRIC_NAMES)}
        metrics_std = {name: float(err[i, 1].item()) for i, name in enumerate(METRIC_NAMES)}
        sample = {
            "seed": seed,
            "elapsed_s": t1 - t0,
            "metrics_mean": metrics_mean,
            "metrics_std": metrics_std,
            "error_tensor": err,
            "pose_p": pose_p.cpu(),
            "tran_p": tran_p.cpu(),
        }
        samples.append(sample)
        current_msg = format_live_seed_message(seq_idx, seed, t1 - t0, metrics_mean)
        print(f"\r{current_msg}", end="", flush=True)
    print()
    return samples


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    if args.out_dir is None:
        suffix = "all" if args.all_seqs else "first_seq"
        ddim_tag = f"_ddim{args.ddim}" if args.ddim is not None else ""
        chunk_tag = f"_chunk{args.chunk_size}" if args.chunk_size is not None else ""
        known_gt_tag = f"_gt{args.known_gt_frames}" if args.known_gt_frames > 0 else ""
        skip_tag = f"_skip{args.skip_first_frames}" if args.skip_first_frames > 0 else ""
        out_dir = (
            ckpt.parent.parent
            / f"eval_imuposer_streaming_bestofk_{suffix}_k{args.k}_{args.selection}{ddim_tag}{chunk_tag}{known_gt_tag}{skip_tag}"
        )
    else:
        out_dir = Path(args.out_dir)
    seq_dir = out_dir / "sequences"
    vis_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_with_overrides(
        str(ckpt),
        args.exp,
        ddim_steps=args.ddim,
        chunk_size=args.chunk_size,
        history_frames=args.history_frames,
        sampler=args.sampler,
    )
    evaluator, body_model = load_pose_evaluator()
    data = torch.load(args.eval_pt, map_location="cpu")

    num_sequences = len(data["acc"]) if args.all_seqs else 1
    sequence_results = []
    selected_best_errors = []
    all_start = time.perf_counter()

    for seq_idx in tqdm(range(num_sequences), desc="sequences"):
        acc5 = data["acc"][seq_idx].float()
        ori5 = data["ori"][seq_idx].float()
        pose_t = data["pose"][seq_idx].float()
        tran_t = data["tran"][seq_idx].float()

        samples = evaluate_sequence(
            model,
            evaluator,
            seq_idx,
            acc5,
            ori5,
            pose_t,
            tran_t,
            args.combo,
            args.window,
            args.k,
            args.seed_start,
            known_gt_frames=args.known_gt_frames,
            skip_first_frames=args.skip_first_frames,
            inject_observed_prefix_into_output=args.inject_observed_prefix_into_output,
            init_rollout_from_observed_prefix=args.init_rollout_from_observed_prefix,
        )

        best_selected = pick_best_sample(samples, args.selection)
        if args.selection == "composite":
            best_selected_summary = {"seed": best_selected["seed"], "composite_z": best_selected["composite_z"]}
        else:
            best_selected_summary = {
                "seed": best_selected["seed"],
                "selection_value": best_selected["metrics_mean"][args.selection],
            }

        seq_payload = {
            "sequence_index": seq_idx,
            "selection": args.selection,
            "known_gt_frames": args.known_gt_frames,
            "skip_first_frames": args.skip_first_frames,
            "best_selected_seed": best_selected["seed"],
            "best_selected_metrics_mean": best_selected["metrics_mean"],
            "best_selected_metrics_std": best_selected["metrics_std"],
            "samples": [
                {
                    "seed": sample["seed"],
                    "elapsed_s": sample["elapsed_s"],
                    "metrics_mean": sample["metrics_mean"],
                    "metrics_std": sample["metrics_std"],
                    **({"composite_z": sample["composite_z"]} if "composite_z" in sample else {}),
                }
                for sample in samples
            ],
            **best_selected_summary,
        }
        sequence_results.append(seq_payload)
        selected_best_errors.append(best_selected["error_tensor"])

        torch.save(
            {
                "pose_p": best_selected["pose_p"],
                "pose_t": pose_t.cpu(),
                "tran_p": best_selected["tran_p"],
                "tran_t": tran_t.cpu(),
                "summary": seq_payload,
            },
            seq_dir / f"{seq_idx + 1}.pt",
        )

        if seq_idx < args.render_best_count:
            render_side_by_side(
                body_model,
                pose_t,
                torch.zeros_like(tran_t),
                best_selected["pose_p"],
                torch.zeros_like(best_selected["tran_p"]),
                vis_dir / f"{seq_idx + 1}.mp4",
                backend=args.render_backend,
            )

        print(
            "BEST_SELECTED",
            {
                "sequence_index": seq_idx,
                "seed": best_selected["seed"],
                **format_metric_dict(best_selected["metrics_mean"]),
            },
        )
        sys.stdout.flush()

    selected_summary = summarize_errors(selected_best_errors)

    final_summary = {
        "ckpt": str(ckpt),
        "exp": args.exp,
        "eval_pt": args.eval_pt,
        "combo": args.combo,
        "window": args.window,
        "ddim": args.ddim,
        "chunk_size": args.chunk_size,
        "history_frames": args.history_frames,
        "k": args.k,
        "seed_start": args.seed_start,
        "selection": args.selection,
        "known_gt_frames": args.known_gt_frames,
        "skip_first_frames": args.skip_first_frames,
        "num_sequences": num_sequences,
        "total_elapsed_s": time.perf_counter() - all_start,
        "sequence_results": sequence_results,
        "selected_best_summary": selected_summary,
    }

    torch.save(final_summary, out_dir / "summary.pt")
    torch.save({"errors": torch.stack(selected_best_errors), "summary": selected_summary}, out_dir / "selected_best_metrics.pt")
    save_summary_text(out_dir / "selected_best_metrics.txt", selected_summary)

    print("SUMMARY_PATH", str(out_dir / "summary.pt"))
    print("SELECTED_BEST_SUMMARY")
    for idx, name in enumerate(METRIC_NAMES):
        print(
            {
                "metric": name,
                "mean": round(selected_summary[idx, 0].item(), 4),
                "std": round(selected_summary[idx, 1].item(), 4),
            }
        )


if __name__ == "__main__":
    main()
