#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import time
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.pure_motion.imu_utils import (
    DEFAULT_SENSOR_COMBOS,
    build_f_imu,
    build_f_imu_selected,
)
from scripts.eval.eval_imu_streaming import (
    load_model,
    stream_predict_sequence_causal,
    stream_predict_sequence_causal_direct,
    stream_predict_sequence_causal_fast,
    stream_predict_sequence_causal_gpu,
)


IMUPOSER_DEVICE_TO_SLOT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark causal IMU streaming speed")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/root/autodl-tmp/dataset/processed/eval/imuposer_test.pt",
    )
    parser.add_argument("--combo", type=str, default="lw_rp_h")
    parser.add_argument("--seq-idx", type=int, default=0)
    parser.add_argument("--window", type=int, default=120)
    return parser.parse_args()


def map_imuposer_to_model_input(acc5, ori5, model, combo_name):
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

    raise ValueError(f"Unsupported imu_dim for benchmark: {imu_dim}")


def benchmark_once(fn, model, f_imu, sensor_mask, window):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pose, tran = fn(model, f_imu, sensor_mask, window)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return {
        "pose_shape": tuple(pose.shape),
        "tran_shape": tuple(tran.shape),
        "total_s": elapsed,
        "ms_per_frame": elapsed * 1000.0 / f_imu.shape[0],
        "fps": f_imu.shape[0] / elapsed,
    }


def main():
    args = parse_args()
    model = load_model(args.ckpt, args.exp)
    data = torch.load(args.eval_pt, map_location="cpu")
    acc = data["acc"][args.seq_idx].float()
    ori = data["ori"][args.seq_idx].float()
    f_imu, sensor_mask = map_imuposer_to_model_input(acc, ori, model, args.combo)

    runs = [
        ("stage0_reference", lambda m, f, s, w: stream_predict_sequence_causal(m, f, s)),
        ("stage1_direct_forward", lambda m, f, s, w: stream_predict_sequence_causal_direct(m, f, s)),
        ("stage2_gpu_history_rollout", lambda m, f, s, w: stream_predict_sequence_causal_gpu(m, f, s)),
        ("stage3_reuse_buffers", lambda m, f, s, w: stream_predict_sequence_causal_fast(m, f, s)),
    ]

    results = []
    baseline = None
    for name, fn in runs:
        result = benchmark_once(fn, model, f_imu, sensor_mask, args.window)
        result["name"] = name
        if baseline is None:
            baseline = result["ms_per_frame"]
            result["speedup_vs_baseline"] = 1.0
        else:
            result["speedup_vs_baseline"] = baseline / result["ms_per_frame"]
        results.append(result)

    for result in results:
        print(
            {
                "name": result["name"],
                "total_s": round(result["total_s"], 4),
                "ms_per_frame": round(result["ms_per_frame"], 4),
                "fps": round(result["fps"], 4),
                "speedup_vs_baseline": round(result["speedup_vs_baseline"], 4),
                "pose_shape": result["pose_shape"],
                "tran_shape": result["tran_shape"],
            }
        )


if __name__ == "__main__":
    main()
