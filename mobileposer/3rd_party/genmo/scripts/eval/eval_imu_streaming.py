#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm
from scripts.eval.aitviewer_render import render_smpl_side_by_side_aitviewer

from gem.datasets.pure_motion.imu_utils import DEFAULT_SENSOR_COMBOS, build_f_imu
from gem.utils.cam_utils import compute_bbox_info_bedlam
from gem.utils.geo_transform import normalize_kp2d
from gem.utils.motion_utils import init_rollout_w_Rt_state, rollout_step_w_Rt
from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


WINDOW_FRAMES = 120
IMUPOSER_DEVICE_TO_SLOT = {
    0: 0,  # left wrist
    1: 1,  # right wrist
    2: 2,  # left phone -> left thigh / pocket
    3: 3,  # right phone -> right thigh / pocket
    4: 4,  # head
}


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming IMU evaluation for GEM IMU model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/project/GENMO/outputs/gem_imu_amass/gem_imu_/version_10/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/root/autodl-tmp/dataset/processed/eval/imuposer_test.pt",
    )
    parser.add_argument("--exp", type=str, default="gem_imu")
    parser.add_argument("--combo", type=str, default="lw_rp_h")
    parser.add_argument("--window", type=int, default=WINDOW_FRAMES)
    parser.add_argument("--render-count", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to sibling directory next to checkpoint: eval_imuposer_streaming",
    )
    return parser.parse_args()


def load_pose_evaluator():
    repo_root = Path("/home/project/GENMO")
    mp_root = repo_root / "3rdparty" / "mobileposer"
    sys.path.insert(0, str(mp_root))
    from config import joint_set, paths, datasets  # type: ignore

    paths.smpl_file = mp_root / "smpl" / "basicmodel_m.pkl"
    import articulate as art  # type: ignore

    body_model = art.ParametricModel(paths.smpl_file)
    
    class PoseEvaluator:
        def __init__(self):
            self._eval_fn = art.FullMotionEvaluator(
                paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=datasets.fps
            )

        def eval(self, pose_p, pose_t, joint_p=None, tran_p=None, tran_t=None):
            pose_p = pose_p.clone().view(-1, 24, 3, 3)
            pose_t = pose_t.clone().view(-1, 24, 3, 3)
            if tran_p is not None and tran_t is not None:
                tran_p = tran_p.clone().view(-1, 3)
                tran_t = tran_t.clone().view(-1, 3)
            else:
                tran_p = torch.zeros(pose_p.shape[0], 3, device=pose_p.device)
                tran_t = torch.zeros(pose_t.shape[0], 3, device=pose_t.device)

            ignored = torch.tensor(joint_set.ignored, device=pose_p.device, dtype=torch.long)
            identity_p = (
                torch.eye(3, device=pose_p.device)
                .view(1, 1, 3, 3)
                .expand(pose_p.shape[0], len(joint_set.ignored), 3, 3)
            )
            identity_t = (
                torch.eye(3, device=pose_t.device)
                .view(1, 1, 3, 3)
                .expand(pose_t.shape[0], len(joint_set.ignored), 3, 3)
            )
            pose_p = pose_p.index_copy(1, ignored, identity_p)
            pose_t = pose_t.index_copy(1, ignored.to(pose_t.device), identity_t)

            errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
            return torch.stack(
                [
                    errs[9],
                    errs[3],
                    errs[9],
                    errs[0] * 100,
                    errs[7] * 100,
                    errs[1] * 100,
                    errs[4] / 100,
                    errs[6],
                ]
            )

        @staticmethod
        def print(errors):
            names = [
                "SIP Error (deg)",
                "Angular Error (deg)",
                "Masked Angular Error (deg)",
                "Positional Error (cm)",
                "Masked Positional Error (cm)",
                "Mesh Error (cm)",
                "Jitter Error (100m/s^3)",
                "Distance Error (cm)",
            ]
            for idx, name in enumerate(names):
                mean = errors[idx, 0].item()
                std = errors[idx, 1].item()
                print(f"{name}: {mean:.2f} (+/- {std:.2f})")

    evaluator = PoseEvaluator()
    return evaluator, body_model


def resolve_output_dir(args):
    if args.output_dir is not None:
        return Path(args.output_dir)
    ckpt_dir = Path(args.ckpt).resolve().parent.parent
    return ckpt_dir / "eval_imuposer_streaming"


def load_model(ckpt_path, exp_name):
    config_dir = Path("/home/project/GENMO/configs")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = hydra.compose(config_name="train", overrides=[f"exp={exp_name}", "use_wandb=false"])
    model = instantiate(cfg.model, _recursive_=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.pipeline.args.use_cfg_sampler_for_gen = False
    model.pipeline.denoiser3d.args.use_cfg_sampler_for_gen = False
    model = model.cuda().eval()
    return model


def body_params_to_full_pose(body_pose_aa, global_orient_aa):
    if body_pose_aa.ndim == 1:
        body_pose_aa = body_pose_aa[None]
    if global_orient_aa.ndim == 1:
        global_orient_aa = global_orient_aa[None]
    length = body_pose_aa.shape[0]
    full_pose = torch.eye(3, device=body_pose_aa.device, dtype=body_pose_aa.dtype).view(1, 1, 3, 3).repeat(length, 24, 1, 1)
    full_pose[:, 0] = axis_angle_to_matrix(global_orient_aa)
    full_pose[:, 1:22] = axis_angle_to_matrix(body_pose_aa.view(length, 21, 3))
    return full_pose


def map_imuposer_to_f_imu(acc5, ori5, combo_name):
    length = acc5.shape[0]
    acc7 = torch.zeros(length, 7, 3, dtype=torch.float32)
    ori7 = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(length, 7, 1, 1)
    for src_idx, dst_idx in IMUPOSER_DEVICE_TO_SLOT.items():
        acc7[:, dst_idx] = acc5[:, src_idx]
        ori7[:, dst_idx] = ori5[:, src_idx]
    sensor_mask = torch.zeros(7, dtype=torch.float32)
    sensor_mask[DEFAULT_SENSOR_COMBOS[combo_name]] = 1.0
    for slot_idx in range(7):
        if sensor_mask[slot_idx] == 0:
            acc7[:, slot_idx] = 0
            ori7[:, slot_idx] = 0
    f_imu, _ = build_f_imu(acc7, ori7, sensor_mask, include_combo_mask=True)
    return f_imu, sensor_mask


def make_window_data(f_imu_window, sensor_mask):
    length = f_imu_window.shape[0]
    zeros_bool = torch.zeros(length, dtype=torch.bool)
    zeros_mask = {
        "valid": torch.ones(length, dtype=torch.bool),
        "humanoid": torch.zeros(length, dtype=torch.bool),
        "has_img_mask": zeros_bool.clone(),
        "has_2d_mask": zeros_bool.clone(),
        "has_cam_mask": zeros_bool.clone(),
        "has_audio_mask": zeros_bool.clone(),
        "has_music_mask": zeros_bool.clone(),
        "has_imu_mask": torch.ones(length, dtype=torch.bool),
        "2d_only": False,
        "vitpose": False,
        "bbx_xys": False,
        "f_imgseq": False,
        "spv_incam_only": False,
        "invalid_contact": False,
    }
    return {
        "meta": [{"mode": "default"}],
        "length": torch.tensor(length, dtype=torch.long),
        "kp2d": torch.zeros(length, 17, 3, dtype=torch.float32),
        "bbx_xys": torch.zeros(length, 3, dtype=torch.float32),
        "K_fullimg": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1),
        "cam_angvel": torch.zeros(length, 6, dtype=torch.float32),
        "cam_tvel": torch.zeros(length, 3, dtype=torch.float32),
        "f_imgseq": torch.zeros(length, 1024, dtype=torch.float32),
        "f_imu": f_imu_window,
        "mask": zeros_mask,
        "caption": "",
        "has_text": torch.tensor(False),
        "imu_sensor_mask": sensor_mask,
    }


def _ensure_cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _init_streaming_runtime(model, max_len, f_imu_dim, device):
    if model.endecoder.obs_indices_dict is None:
        model.endecoder.build_obs_indices_dict()

    kp2d = torch.zeros(1, max_len, 17, 3, device=device)
    bbx_xys = torch.zeros(1, max_len, 3, device=device)
    K_fullimg = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, max_len, 1, 1)
    cam_angvel = torch.zeros(1, max_len, 6, device=device)
    cam_tvel = torch.zeros(1, max_len, 3, device=device)
    f_imgseq = torch.zeros(1, max_len, 1024, device=device)
    false_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)

    runtime = {
        "device": device,
        "max_len": max_len,
        "motion_dim": model.endecoder.get_motion_dim(),
        "sample_indices_dict": model.endecoder.obs_indices_dict,
        "length": torch.zeros(1, dtype=torch.long, device=device),
        "obs": normalize_kp2d(kp2d, bbx_xys),
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_cam_angvel": cam_angvel.clone(),
        "cam_tvel": cam_tvel,
        "f_imgseq": f_imgseq,
        "target_x": torch.zeros(1, max_len, model.endecoder.get_motion_dim(), device=device),
        "f_imu": torch.zeros(1, max_len, f_imu_dim, device=device),
        "observed_motion_3d": torch.zeros(1, max_len, model.endecoder.get_motion_dim(), device=device),
        "motion_mask_3d": torch.zeros(1, max_len, model.endecoder.get_motion_dim(), device=device),
        "f_cliffcam": compute_bbox_info_bedlam(bbx_xys, K_fullimg),
        "condition_mask": {
            "has_img_mask": false_mask.clone(),
            "has_2d_mask": false_mask.clone(),
            "has_cam_mask": false_mask.clone(),
            "has_audio_mask": false_mask.clone(),
            "has_music_mask": false_mask.clone(),
            "has_imu_mask": false_mask.clone(),
            "j2d_visible_mask": false_mask[:, :, None].repeat(1, 1, 17),
        },
        "meta": [{"mode": "default"}],
        "caption": [""],
        "has_text": torch.zeros(1, dtype=torch.bool, device=device),
        "encoded_text": None,
    }
    runtime["encoded_text"] = model.encode_text(runtime["caption"], runtime["has_text"])

    if model.model_cfg.normalize_cam_angvel:
        runtime["f_cam_angvel"] = (runtime["f_cam_angvel"] - model.cam_angvel_mean) / model.cam_angvel_std
    for key in model.normalizer_stats:
        if key in runtime:
            runtime[key] = model.normalize_attr(runtime[key], key)
    return runtime


def _make_causal_chunk_batch_direct(model, f_imu_chunk, observed_motion, motion_mask, sensor_mask):
    device = next(model.parameters()).device
    if model.endecoder.obs_indices_dict is None:
        model.endecoder.build_obs_indices_dict()

    L = f_imu_chunk.shape[0]
    kp2d = torch.zeros(1, L, 17, 3, device=device)
    bbx_xys = torch.zeros(1, L, 3, device=device)
    K_fullimg = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, L, 1, 1)
    cam_angvel = torch.zeros(1, L, 6, device=device)
    cam_tvel = torch.zeros(1, L, 3, device=device)
    f_imgseq = torch.zeros(1, L, 1024, device=device)
    false_mask = torch.zeros(1, L, dtype=torch.bool, device=device)

    batch = {
        "length": torch.tensor([L], dtype=torch.long, device=device),
        "obs": normalize_kp2d(kp2d, bbx_xys),
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_cam_angvel": cam_angvel.clone(),
        "cam_tvel": cam_tvel,
        "f_imgseq": f_imgseq,
        "has_text": torch.zeros(1, dtype=torch.bool, device=device),
        "B": 1,
        "L": L,
        "mode": "default",
        "target_x": torch.zeros(1, L, model.endecoder.get_motion_dim(), device=device),
        "sample_indices_dict": model.endecoder.obs_indices_dict,
        "f_imu": f_imu_chunk[None].to(device),
        "device": device,
        "meta": [{"mode": "default"}],
        "caption": [""],
        "has_humanoid_data": torch.ones(1, dtype=torch.bool, device=device),
        "observed_motion_3d": observed_motion[None].to(device),
        "motion_mask_3d": motion_mask[None].to(device),
        "encoded_text": None,
    }
    batch["encoded_text"] = model.encode_text(batch["caption"], batch["has_text"])
    batch["f_cliffcam"] = compute_bbox_info_bedlam(batch["bbx_xys"], batch["K_fullimg"])
    batch["condition_mask"] = {
        "has_img_mask": false_mask.clone(),
        "has_2d_mask": false_mask.clone(),
        "has_cam_mask": false_mask.clone(),
        "has_audio_mask": false_mask.clone(),
        "has_music_mask": false_mask.clone(),
        "has_imu_mask": torch.ones(1, L, dtype=torch.bool, device=device),
        "j2d_visible_mask": false_mask[:, :, None].repeat(1, 1, 17),
    }

    if model.model_cfg.normalize_cam_angvel:
        batch["f_cam_angvel"] = (batch["f_cam_angvel"] - model.cam_angvel_mean) / model.cam_angvel_std
    for key in model.normalizer_stats:
        if key in batch:
            batch[key] = model.normalize_attr(batch[key], key)
    return model.create_condition_mask(batch, cond_mask_cfg=None, mode=None, train=False)


def _make_causal_chunk_batch_reuse(model, runtime, f_imu_chunk, history_len, history_latents):
    device = runtime["device"]
    L = f_imu_chunk.shape[0]
    future_len = L - history_len
    batch = {
        "length": runtime["length"],
        "obs": runtime["obs"][:, :L],
        "bbx_xys": runtime["bbx_xys"][:, :L],
        "K_fullimg": runtime["K_fullimg"][:, :L],
        "cam_angvel": runtime["cam_angvel"][:, :L],
        "f_cam_angvel": runtime["f_cam_angvel"][:, :L],
        "cam_tvel": runtime["cam_tvel"][:, :L],
        "f_imgseq": runtime["f_imgseq"][:, :L],
        "has_text": runtime["has_text"],
        "B": 1,
        "L": L,
        "mode": "default",
        "target_x": runtime["target_x"][:, :L],
        "sample_indices_dict": runtime["sample_indices_dict"],
        "f_imu": runtime["f_imu"][:, :L],
        "device": device,
        "meta": runtime["meta"],
        "caption": runtime["caption"],
        "has_humanoid_data": torch.ones(1, dtype=torch.bool, device=device),
        "observed_motion_3d": runtime["observed_motion_3d"][:, :L],
        "motion_mask_3d": runtime["motion_mask_3d"][:, :L],
        "encoded_text": runtime["encoded_text"],
        "f_cliffcam": runtime["f_cliffcam"][:, :L],
        "condition_mask": {
            "has_img_mask": runtime["condition_mask"]["has_img_mask"][:, :L],
            "has_2d_mask": runtime["condition_mask"]["has_2d_mask"][:, :L],
            "has_cam_mask": runtime["condition_mask"]["has_cam_mask"][:, :L],
            "has_audio_mask": runtime["condition_mask"]["has_audio_mask"][:, :L],
            "has_music_mask": runtime["condition_mask"]["has_music_mask"][:, :L],
            "has_imu_mask": runtime["condition_mask"]["has_imu_mask"][:, :L],
            "j2d_visible_mask": runtime["condition_mask"]["j2d_visible_mask"][:, :L],
        },
    }
    batch["length"][0] = L
    batch["f_imu"].zero_()
    batch["f_imu"][0, :L] = f_imu_chunk.to(device)
    batch["target_x"].zero_()
    batch["observed_motion_3d"].zero_()
    batch["motion_mask_3d"].zero_()
    batch["condition_mask"]["has_imu_mask"].zero_()
    batch["condition_mask"]["has_imu_mask"][:, :L] = True
    if history_len > 0:
        batch["observed_motion_3d"][0, :history_len] = history_latents
        batch["motion_mask_3d"][0, :history_len] = 1.0
    return model.create_condition_mask(batch, cond_mask_cfg=None, mode=None, train=False)


@torch.no_grad()
def decode_window_batch(model, windows, sensor_mask):
    from gem.utils.cam_utils import compute_bbox_info_bedlam
    from gem.utils.geo_transform import normalize_kp2d

    if model.endecoder.obs_indices_dict is None:
        model.endecoder.build_obs_indices_dict()

    B, L, _ = windows.shape
    device = next(model.parameters()).device
    kp2d = torch.zeros(B, L, 17, 3, device=device)
    bbx_xys = torch.zeros(B, L, 3, device=device)
    K_fullimg = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, L, 1, 1)
    cam_angvel = torch.zeros(B, L, 6, device=device)
    cam_tvel = torch.zeros(B, L, 3, device=device)
    f_imgseq = torch.zeros(B, L, 1024, device=device)
    has_text = torch.zeros(B, dtype=torch.bool, device=device)

    batch = {
        "length": torch.full((B,), L, dtype=torch.long, device=device),
        "obs": normalize_kp2d(kp2d, bbx_xys),
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_cam_angvel": cam_angvel.clone(),
        "cam_tvel": cam_tvel,
        "f_imgseq": f_imgseq,
        "has_text": has_text,
        "B": B,
        "L": L,
        "mode": "default",
        "target_x": torch.zeros(B, L, model.endecoder.get_motion_dim(), device=device),
        "sample_indices_dict": model.endecoder.obs_indices_dict,
        "f_imu": windows.to(device),
        "device": device,
        "meta": [{"mode": "default"} for _ in range(B)],
        "caption": [""] * B,
    }
    batch["encoded_text"] = model.encode_text(batch["caption"], batch["has_text"])
    batch["f_cliffcam"] = compute_bbox_info_bedlam(batch["bbx_xys"], batch["K_fullimg"]).to(device)

    false_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    condition_mask = {
        "has_img_mask": false_mask.clone(),
        "has_2d_mask": false_mask.clone(),
        "has_cam_mask": false_mask.clone(),
        "has_audio_mask": false_mask.clone(),
        "has_music_mask": false_mask.clone(),
        "has_imu_mask": torch.ones(B, L, dtype=torch.bool, device=device),
        "j2d_visible_mask": false_mask[:, :, None].repeat(1, 1, 17),
    }
    batch["condition_mask"] = condition_mask

    if model.model_cfg.normalize_cam_angvel:
        batch["f_cam_angvel"] = (batch["f_cam_angvel"] - model.cam_angvel_mean) / model.cam_angvel_std
    for k in model.normalizer_stats:
        if k in batch:
            batch[k] = model.normalize_attr(batch[k], k)

    batch = model.create_condition_mask(batch, cond_mask_cfg=None, mode=None, train=False)
    outputs = model.pipeline.forward(
        batch,
        train=False,
        postproc=False,
        static_cam=True,
        test_mode="default",
    )
    decode = outputs["decode_dict"]
    result = {
        "body_pose": decode["body_pose"][:, -1].detach().cpu(),
        "global_orient_c": decode["global_orient"][:, -1].detach().cpu(),
        "global_orient_gv": decode["global_orient_gv"][:, -1].detach().cpu(),
        "local_transl_vel_curr": decode["local_transl_vel"][:, -1].detach().cpu(),
        "local_transl_vel_prev": (
            decode["local_transl_vel"][:, -2].detach().cpu() if L > 1 else None
        ),
    }
    if "betas" in decode:
        result["betas"] = decode["betas"][:, -1].detach().cpu()
    return result


@torch.no_grad()
def stream_predict_sequence_legacy(model, f_imu_seq, sensor_mask, window, batch_size=16):
    pred_pose = []
    pred_tran = []
    windows = []
    rollout_state = None
    for frame_idx in range(f_imu_seq.shape[0]):
        left = max(0, frame_idx - window + 1)
        chunk = f_imu_seq[left : frame_idx + 1]
        if chunk.shape[0] < window:
            pad = chunk[:1].repeat(window - chunk.shape[0], 1)
            chunk = torch.cat([pad, chunk], dim=0)
        windows.append(chunk)

    for start in tqdm(range(0, len(windows), batch_size), leave=False):
        batch_windows = torch.stack(windows[start : start + batch_size], dim=0)
        with contextlib.redirect_stdout(io.StringIO()):
            decoded = decode_window_batch(model, batch_windows, sensor_mask)
        chunk_size = batch_windows.shape[0]
        for i in range(chunk_size):
            gv_curr = decoded["global_orient_gv"][i]
            gc_curr = decoded["global_orient_c"][i]
            lv_prev = decoded["local_transl_vel_prev"][i] if decoded["local_transl_vel_prev"] is not None else None
            lv_curr = decoded["local_transl_vel_curr"][i]
            if rollout_state is None:
                rollout_state = init_rollout_w_Rt_state(
                    gv_curr.cuda(), gc_curr.cuda(), device=model.device
                )
            body_params_global, rollout_state = rollout_step_w_Rt(
                rollout_state,
                global_orient_gv_curr=gv_curr.cuda(),
                global_orient_c_curr=gc_curr.cuda(),
                cam_angvel_prev=None,
                local_transl_vel_prev=(lv_prev.cuda() if lv_prev is not None else None),
                local_transl_vel_curr=(None if lv_prev is not None else lv_curr.cuda()),
            )
            pose_frame = body_params_to_full_pose(
                decoded["body_pose"][i].cuda(),
                body_params_global["global_orient"][0],
            ).cpu()[0]
            transl_frame = body_params_global["transl"][0].detach().cpu()
            pred_pose.append(pose_frame)
            pred_tran.append(transl_frame)
    return torch.stack(pred_pose), torch.stack(pred_tran)


def _rollout_decoded_frames(model, decoded, start_idx, end_idx, rollout_state, pred_pose, pred_tran):
    for i in range(start_idx, end_idx):
        gv_curr = decoded["global_orient_gv"][i]
        gc_curr = decoded["global_orient_c"][i]
        lv_prev = decoded["local_transl_vel_prev"][i] if decoded["local_transl_vel_prev"] is not None else None
        lv_curr = decoded["local_transl_vel_curr"][i]
        if rollout_state is None:
            rollout_state = init_rollout_w_Rt_state(
                gv_curr.cuda(), gc_curr.cuda(), device=model.device
            )
        body_params_global, rollout_state = rollout_step_w_Rt(
            rollout_state,
            global_orient_gv_curr=gv_curr.cuda(),
            global_orient_c_curr=gc_curr.cuda(),
            cam_angvel_prev=None,
            local_transl_vel_prev=(lv_prev.cuda() if lv_prev is not None else None),
            local_transl_vel_curr=(None if lv_prev is not None else lv_curr.cuda()),
        )
        pose_frame = body_params_to_full_pose(
            decoded["body_pose"][i].cuda(),
            body_params_global["global_orient"][0],
        ).cpu()[0]
        transl_frame = body_params_global["transl"][0].detach().cpu()
        pred_pose.append(pose_frame)
        pred_tran.append(transl_frame)
    return rollout_state


def _rollout_decoded_frames_gpu(
    model, decoded, start_idx, end_idx, rollout_state, pred_pose, pred_tran
):
    for i in range(start_idx, end_idx):
        gv_curr = decoded["global_orient_gv"][i]
        gc_curr = decoded["global_orient_c"][i]
        lv_prev = decoded["local_transl_vel_prev"][i] if decoded["local_transl_vel_prev"] is not None else None
        lv_curr = decoded["local_transl_vel_curr"][i]
        if rollout_state is None:
            rollout_state = init_rollout_w_Rt_state(gv_curr, gc_curr, device=model.device)
        body_params_global, rollout_state = rollout_step_w_Rt(
            rollout_state,
            global_orient_gv_curr=gv_curr,
            global_orient_c_curr=gc_curr,
            cam_angvel_prev=None,
            local_transl_vel_prev=lv_prev,
            local_transl_vel_curr=(None if lv_prev is not None else lv_curr),
        )
        pose_frame = body_params_to_full_pose(
            decoded["body_pose"][i],
            body_params_global["global_orient"][0],
        )[0]
        transl_frame = body_params_global["transl"][0]
        pred_pose.append(pose_frame)
        pred_tran.append(transl_frame)
    return rollout_state


def _encode_observed_motion_prefix(model, pose_prefix, tran_prefix):
    if pose_prefix is None or tran_prefix is None:
        return None
    if pose_prefix.shape[0] == 0:
        return None
    device = next(model.parameters()).device
    pose_prefix = pose_prefix.to(device)
    tran_prefix = tran_prefix.to(device)
    latent_len = pose_prefix.shape[0]
    if latent_len == 1:
        pose_prefix = torch.cat([pose_prefix, pose_prefix.clone()], dim=0)
        tran_prefix = torch.cat([tran_prefix, tran_prefix.clone()], dim=0)
    aa = matrix_to_axis_angle(pose_prefix)
    K = pose_prefix.shape[0]
    zeros_betas = torch.zeros(1, K, 10, device=device, dtype=pose_prefix.dtype)
    encode_inputs = {
        "smpl_params_c": {
            "body_pose": aa[:, 1:22].reshape(1, K, 63),
            "betas": zeros_betas,
            "global_orient": aa[:, 0].reshape(1, K, 3),
        },
        "smpl_params_w": {
            "transl": tran_prefix.reshape(1, K, 3),
            "global_orient": aa[:, 0].reshape(1, K, 3),
        },
        "R_c2gv": torch.eye(3, device=device, dtype=pose_prefix.dtype).view(1, 1, 3, 3).repeat(1, K, 1, 1),
    }
    encoded = model.endecoder.encode(encode_inputs)[0].detach()
    return encoded[:latent_len]


def _init_rollout_from_observed_prefix(model, pose_prefix, tran_prefix, pred_pose, pred_tran):
    if pose_prefix is None or tran_prefix is None or pose_prefix.shape[0] == 0:
        return None
    pose_prefix = pose_prefix.to(model.device)
    tran_prefix = tran_prefix.to(model.device)
    for i in range(pose_prefix.shape[0]):
        pred_pose.append(pose_prefix[i].detach().clone())
        pred_tran.append(tran_prefix[i].detach().clone())
    aa = matrix_to_axis_angle(pose_prefix)
    last_global_orient = aa[-1, 0]
    rollout_state = init_rollout_w_Rt_state(
        last_global_orient,
        last_global_orient,
        device=model.device,
    )
    rollout_state["transl_pre"] = tran_prefix[-1:].detach().clone().to(model.device)
    rollout_state["global_orient_pre"] = last_global_orient[None].detach().clone().to(model.device)
    return rollout_state


@torch.no_grad()
def stream_predict_sequence_causal(model, f_imu_seq, sensor_mask, history_frames=None, chunk_size=None):
    motion_dim = model.endecoder.get_motion_dim()
    history_frames = (
        model.streaming_history_frames if history_frames is None else int(history_frames)
    )
    chunk_size = (
        model.streaming_online_chunk_frames if chunk_size is None else int(chunk_size)
    )
    history_frames = max(history_frames, 1)
    chunk_size = max(chunk_size, 1)

    pred_pose = []
    pred_tran = []
    latent_history = []
    rollout_state = None
    start = 0
    while start < f_imu_seq.shape[0]:
        future_len = min(chunk_size, f_imu_seq.shape[0] - start)
        history_len = min(len(latent_history), history_frames)
        left = start - history_len
        right = start + future_len
        f_imu_chunk = f_imu_seq[left:right]
        data = make_window_data(f_imu_chunk, sensor_mask)
        data["has_humanoid_data"] = True

        observed_motion = torch.zeros(f_imu_chunk.shape[0], motion_dim, dtype=torch.float32)
        motion_mask = torch.zeros_like(observed_motion)
        if history_len > 0:
            hist_latents = torch.stack(latent_history[-history_len:], dim=0)
            observed_motion[:history_len] = hist_latents
            motion_mask[:history_len] = 1.0
        data["observed_motion_3d"] = observed_motion
        data["motion_mask_3d"] = motion_mask

        with contextlib.redirect_stdout(io.StringIO()):
            pred = model.predict(data, static_cam=True, postproc=False)
        outputs = pred["net_outputs"]
        decode = outputs["decode_dict"]
        pred_x = outputs["model_output"].get("pred_x", outputs["model_output"].get("pred_x_start"))
        pred_x = pred_x[0].detach().cpu()

        future_start = history_len
        future_end = history_len + future_len
        for i in range(future_start, future_end):
            latent_history.append(pred_x[i])

        decoded = {
            "body_pose": decode["body_pose"][0].detach().cpu(),
            "global_orient_c": decode["global_orient"][0].detach().cpu(),
            "global_orient_gv": decode["global_orient_gv"][0].detach().cpu(),
            "local_transl_vel_curr": decode["local_transl_vel"][0].detach().cpu(),
            "local_transl_vel_prev": None,
        }
        if decode["local_transl_vel"].shape[1] > 1:
            prev = torch.zeros_like(decoded["local_transl_vel_curr"])
            prev[1:] = decode["local_transl_vel"][0, :-1].detach().cpu()
            prev[0] = decode["local_transl_vel"][0, 0].detach().cpu()
            decoded["local_transl_vel_prev"] = prev
        rollout_state = _rollout_decoded_frames(
            model,
            decoded,
            future_start,
            future_end,
            rollout_state,
            pred_pose,
            pred_tran,
        )
        start += future_len
    return torch.stack(pred_pose), torch.stack(pred_tran)


@torch.no_grad()
def stream_predict_sequence_causal_direct(model, f_imu_seq, sensor_mask, history_frames=None, chunk_size=None):
    motion_dim = model.endecoder.get_motion_dim()
    history_frames = (
        model.streaming_history_frames if history_frames is None else int(history_frames)
    )
    chunk_size = (
        model.streaming_online_chunk_frames if chunk_size is None else int(chunk_size)
    )
    history_frames = max(history_frames, 1)
    chunk_size = max(chunk_size, 1)

    pred_pose = []
    pred_tran = []
    latent_history = []
    rollout_state = None
    start = 0
    while start < f_imu_seq.shape[0]:
        future_len = min(chunk_size, f_imu_seq.shape[0] - start)
        history_len = min(len(latent_history), history_frames)
        left = start - history_len
        right = start + future_len
        f_imu_chunk = f_imu_seq[left:right]

        observed_motion = torch.zeros(f_imu_chunk.shape[0], motion_dim, dtype=torch.float32)
        motion_mask = torch.zeros_like(observed_motion)
        if history_len > 0:
            hist_latents = torch.stack(latent_history[-history_len:], dim=0)
            observed_motion[:history_len] = hist_latents
            motion_mask[:history_len] = 1.0

        batch = _make_causal_chunk_batch_direct(
            model, f_imu_chunk, observed_motion, motion_mask, sensor_mask
        )
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = model.pipeline.forward(
                batch,
                train=False,
                postproc=False,
                static_cam=True,
                test_mode="default",
            )

        decode = outputs["decode_dict"]
        pred_x = outputs["model_output"].get("pred_x", outputs["model_output"].get("pred_x_start"))
        pred_x = pred_x[0].detach().cpu()

        future_start = history_len
        future_end = history_len + future_len
        for i in range(future_start, future_end):
            latent_history.append(pred_x[i])

        decoded = {
            "body_pose": decode["body_pose"][0].detach().cpu(),
            "global_orient_c": decode["global_orient"][0].detach().cpu(),
            "global_orient_gv": decode["global_orient_gv"][0].detach().cpu(),
            "local_transl_vel_curr": decode["local_transl_vel"][0].detach().cpu(),
            "local_transl_vel_prev": None,
        }
        if decode["local_transl_vel"].shape[1] > 1:
            prev = torch.zeros_like(decoded["local_transl_vel_curr"])
            prev[1:] = decode["local_transl_vel"][0, :-1].detach().cpu()
            prev[0] = decode["local_transl_vel"][0, 0].detach().cpu()
            decoded["local_transl_vel_prev"] = prev
        rollout_state = _rollout_decoded_frames(
            model,
            decoded,
            future_start,
            future_end,
            rollout_state,
            pred_pose,
            pred_tran,
        )
        start += future_len
    return torch.stack(pred_pose), torch.stack(pred_tran)


@torch.no_grad()
def stream_predict_sequence_causal_gpu(model, f_imu_seq, sensor_mask, history_frames=None, chunk_size=None):
    history_frames = (
        model.streaming_history_frames if history_frames is None else int(history_frames)
    )
    chunk_size = (
        model.streaming_online_chunk_frames if chunk_size is None else int(chunk_size)
    )
    history_frames = max(history_frames, 1)
    chunk_size = max(chunk_size, 1)

    device = next(model.parameters()).device
    f_imu_seq = f_imu_seq.to(device)
    sensor_mask = sensor_mask.to(device)
    pred_pose = []
    pred_tran = []
    latent_history = []
    rollout_state = None
    start = 0
    while start < f_imu_seq.shape[0]:
        future_len = min(chunk_size, f_imu_seq.shape[0] - start)
        history_len = min(len(latent_history), history_frames)
        left = start - history_len
        right = start + future_len
        f_imu_chunk = f_imu_seq[left:right]

        observed_motion = torch.zeros(
            f_imu_chunk.shape[0], model.endecoder.get_motion_dim(), dtype=torch.float32, device=device
        )
        motion_mask = torch.zeros_like(observed_motion)
        if history_len > 0:
            hist_latents = torch.stack(latent_history[-history_len:], dim=0)
            observed_motion[:history_len] = hist_latents
            motion_mask[:history_len] = 1.0

        batch = _make_causal_chunk_batch_direct(
            model, f_imu_chunk, observed_motion, motion_mask, sensor_mask
        )
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = model.pipeline.forward(
                batch,
                train=False,
                postproc=False,
                static_cam=True,
                test_mode="default",
            )

        decode = outputs["decode_dict"]
        pred_x = outputs["model_output"].get("pred_x", outputs["model_output"].get("pred_x_start"))[0]

        future_start = history_len
        future_end = history_len + future_len
        for i in range(future_start, future_end):
            latent_history.append(pred_x[i].detach())

        decoded = {
            "body_pose": decode["body_pose"][0],
            "global_orient_c": decode["global_orient"][0],
            "global_orient_gv": decode["global_orient_gv"][0],
            "local_transl_vel_curr": decode["local_transl_vel"][0],
            "local_transl_vel_prev": None,
        }
        if decode["local_transl_vel"].shape[1] > 1:
            prev = torch.zeros_like(decoded["local_transl_vel_curr"])
            prev[1:] = decode["local_transl_vel"][0, :-1]
            prev[0] = decode["local_transl_vel"][0, 0]
            decoded["local_transl_vel_prev"] = prev
        rollout_state = _rollout_decoded_frames(
            model,
            decoded,
            future_start,
            future_end,
            rollout_state,
            pred_pose,
            pred_tran,
        )
        start += future_len
    return torch.stack(pred_pose).cpu(), torch.stack(pred_tran).cpu()


@torch.no_grad()
def stream_predict_sequence_causal_fast(
    model,
    f_imu_seq,
    sensor_mask,
    history_frames=None,
    chunk_size=None,
    observed_pose_prefix=None,
    observed_tran_prefix=None,
    inject_observed_prefix_into_output=True,
    init_rollout_from_observed_prefix=True,
    show_progress=False,
    progress_desc=None,
):
    history_frames = (
        model.streaming_history_frames if history_frames is None else int(history_frames)
    )
    chunk_size = (
        model.streaming_online_chunk_frames if chunk_size is None else int(chunk_size)
    )
    history_frames = max(history_frames, 1)
    chunk_size = max(chunk_size, 1)

    device = next(model.parameters()).device
    f_imu_seq = f_imu_seq.to(device)
    sensor_mask = sensor_mask.to(device)
    max_len = min(f_imu_seq.shape[0], history_frames + chunk_size)
    runtime = _init_streaming_runtime(model, max_len, f_imu_seq.shape[-1], device)
    pred_pose = []
    pred_tran = []
    latent_history = []
    observed_latents = _encode_observed_motion_prefix(
        model, observed_pose_prefix, observed_tran_prefix
    )
    if observed_latents is not None:
        latent_history = [observed_latents[i] for i in range(observed_latents.shape[0])]
    if inject_observed_prefix_into_output or init_rollout_from_observed_prefix:
        rollout_state = _init_rollout_from_observed_prefix(
            model,
            observed_pose_prefix if (inject_observed_prefix_into_output or init_rollout_from_observed_prefix) else None,
            observed_tran_prefix if (inject_observed_prefix_into_output or init_rollout_from_observed_prefix) else None,
            pred_pose if inject_observed_prefix_into_output else [],
            pred_tran if inject_observed_prefix_into_output else [],
        )
        if not inject_observed_prefix_into_output:
            pred_pose.clear()
            pred_tran.clear()
            if observed_pose_prefix is not None and observed_tran_prefix is not None and observed_pose_prefix.shape[0] > 0:
                pose_prefix = observed_pose_prefix.to(model.device)
                tran_prefix = observed_tran_prefix.to(model.device)
                aa = matrix_to_axis_angle(pose_prefix)
                last_global_orient = aa[-1, 0]
                rollout_state = init_rollout_w_Rt_state(
                    last_global_orient,
                    last_global_orient,
                    device=model.device,
                )
                rollout_state["transl_pre"] = tran_prefix[-1:].detach().clone().to(model.device)
                rollout_state["global_orient_pre"] = last_global_orient[None].detach().clone().to(model.device)
    else:
        rollout_state = None
    start = len(latent_history)
    if start >= f_imu_seq.shape[0]:
        return torch.stack(pred_pose), torch.stack(pred_tran)
    total_chunks = (int(f_imu_seq.shape[0]) + chunk_size - 1) // chunk_size
    remaining = int(f_imu_seq.shape[0]) - start
    total_chunks = (remaining + chunk_size - 1) // chunk_size
    chunk_iter = range(total_chunks)
    if show_progress:
        chunk_iter = tqdm(chunk_iter, desc=progress_desc or "streaming", leave=False)
    for _ in chunk_iter:
        future_len = min(chunk_size, f_imu_seq.shape[0] - start)
        history_len = min(len(latent_history), history_frames)
        left = start - history_len
        right = start + future_len
        f_imu_chunk = f_imu_seq[left:right]
        history_latents = None
        if history_len > 0:
            history_latents = torch.stack(latent_history[-history_len:], dim=0)
        batch = _make_causal_chunk_batch_reuse(
            model,
            runtime,
            f_imu_chunk,
            history_len,
            history_latents,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = model.pipeline.forward(
                batch,
                train=False,
                postproc=False,
                static_cam=True,
                test_mode="default",
            )

        decode = outputs["decode_dict"]
        pred_x = outputs["model_output"].get("pred_x", outputs["model_output"].get("pred_x_start"))[0]
        future_start = history_len
        future_end = history_len + future_len
        for i in range(future_start, future_end):
            latent_history.append(pred_x[i].detach())

        decoded = {
            "body_pose": decode["body_pose"][0],
            "global_orient_c": decode["global_orient"][0],
            "global_orient_gv": decode["global_orient_gv"][0],
            "local_transl_vel_curr": decode["local_transl_vel"][0],
            "local_transl_vel_prev": None,
        }
        if decode["local_transl_vel"].shape[1] > 1:
            prev = torch.zeros_like(decoded["local_transl_vel_curr"])
            prev[1:] = decode["local_transl_vel"][0, :-1]
            prev[0] = decode["local_transl_vel"][0, 0]
            decoded["local_transl_vel_prev"] = prev
        rollout_state = _rollout_decoded_frames_gpu(
            model,
            decoded,
            future_start,
            future_end,
            rollout_state,
            pred_pose,
            pred_tran,
        )
        start += future_len
    return torch.stack(pred_pose).cpu(), torch.stack(pred_tran).cpu()


@torch.no_grad()
def stream_predict_sequence(
    model,
    f_imu_seq,
    sensor_mask,
    window,
    batch_size=16,
    observed_pose_prefix=None,
    observed_tran_prefix=None,
    inject_observed_prefix_into_output=True,
    init_rollout_from_observed_prefix=True,
    show_progress=False,
    progress_desc=None,
):
    if getattr(model, "enable_causal_streaming", False):
        return stream_predict_sequence_causal_fast(
            model,
            f_imu_seq,
            sensor_mask,
            history_frames=min(window - 1, model.streaming_history_frames),
            chunk_size=model.streaming_online_chunk_frames,
            observed_pose_prefix=observed_pose_prefix,
            observed_tran_prefix=observed_tran_prefix,
            inject_observed_prefix_into_output=inject_observed_prefix_into_output,
            init_rollout_from_observed_prefix=init_rollout_from_observed_prefix,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    return stream_predict_sequence_legacy(model, f_imu_seq, sensor_mask, window, batch_size=batch_size)


def skeleton_to_image(joints, parents, image_wh=(960, 540), color=(50, 90, 220)):
    width, height = image_wh
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    pts = joints[:, [0, 1]].copy()
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-6)
    pts = (pts - center) / scale
    pts[:, 1] *= -1
    pts[:, 0] = pts[:, 0] * (width * 0.65) + width / 2
    pts[:, 1] = pts[:, 1] * (height * 0.65) + height / 2
    pts = pts.astype(np.int32)
    for j, p in enumerate(parents):
        if p is None:
            continue
        cv2.line(canvas, tuple(pts[p]), tuple(pts[j]), color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, tuple(pt), 3, (20, 20, 20), -1, cv2.LINE_AA)
    return canvas


@torch.no_grad()
def render_side_by_side(body_model, pose_t, tran_t, pose_p, tran_p, output_path, fps=30, backend="aitviewer"):
    if backend == "aitviewer":
        try:
            render_smpl_side_by_side_aitviewer(
                body_model,
                pose_t,
                tran_t,
                pose_p,
                tran_p,
                output_path=output_path,
                fps=fps,
            )
            return
        except Exception as exc:
            print(f"[AITViewer render fallback] {exc}")

    _, gt_joints = body_model.forward_kinematics(pose_t, tran=tran_t, calc_mesh=False)
    _, pred_joints = body_model.forward_kinematics(pose_p, tran=tran_p, calc_mesh=False)
    gt_joints = gt_joints.cpu().numpy()
    pred_joints = pred_joints.cpu().numpy()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (1920, 540))
    parents = body_model.parent
    for frame_idx in range(min(len(gt_joints), len(pred_joints))):
        gt_img = skeleton_to_image(gt_joints[frame_idx], parents, color=(55, 125, 235))
        pred_img = skeleton_to_image(pred_joints[frame_idx], parents, color=(235, 110, 55))
        cv2.putText(gt_img, "GT", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(pred_img, "Pred", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
        frame = np.concatenate([gt_img, pred_img], axis=1)
        writer.write(frame)
    writer.release()


def summarize_errors(errors):
    stacked = torch.stack(errors)
    return stacked.mean(dim=0)


def main():
    args = parse_args()
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = out_dir / "sequences"
    vis_dir = out_dir / "videos"
    seq_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, args.exp)
    evaluator, body_model = load_pose_evaluator()
    test_data = torch.load(args.eval_pt, map_location="cpu")

    all_errors = []
    for seq_idx in range(len(test_data["acc"])):
        acc = test_data["acc"][seq_idx].float()
        ori = test_data["ori"][seq_idx].float()
        pose_t = test_data["pose"][seq_idx].float()
        tran_t = test_data["tran"][seq_idx].float()

        f_imu, sensor_mask = map_imuposer_to_f_imu(acc, ori, args.combo)
        pose_p, tran_p = stream_predict_sequence(model, f_imu, sensor_mask, args.window)

        err = evaluator.eval(pose_p.cuda(), pose_t.cuda(), tran_p=tran_p.cuda(), tran_t=tran_t.cuda()).cpu()
        all_errors.append(err)

        torch.save(
            {
                "pose_p": pose_p.cpu(),
                "pose_t": pose_t.cpu(),
                "tran_p": tran_p.cpu(),
                "tran_t": tran_t.cpu(),
            },
            seq_dir / f"{seq_idx + 1}.pt",
        )

        if seq_idx < args.render_count:
            render_side_by_side(body_model, pose_t, tran_t, pose_p, tran_p, vis_dir / f"{seq_idx + 1}.mp4")

    summary = summarize_errors(all_errors)
    torch.save({"errors": torch.stack(all_errors), "summary": summary}, out_dir / "metrics.pt")
    evaluator.print(summary)
    with open(out_dir / "metrics.txt", "w") as handle:
        names = [
            "SIP Error (deg)",
            "Angular Error (deg)",
            "Masked Angular Error (deg)",
            "Positional Error (cm)",
            "Masked Positional Error (cm)",
            "Mesh Error (cm)",
            "Jitter Error (100m/s^3)",
            "Distance Error (cm)",
        ]
        for idx, name in enumerate(names):
            mean = summary[idx, 0].item()
            std = summary[idx, 1].item()
            handle.write(f"{name}: {mean:.4f} +/- {std:.4f}\n")


if __name__ == "__main__":
    main()
