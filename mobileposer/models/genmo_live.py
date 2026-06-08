from __future__ import annotations

import contextlib
import io
import sys
from collections import deque
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate


GENMO_ROOT = Path(__file__).resolve().parents[1] / "3rd_party" / "genmo"
if str(GENMO_ROOT) not in sys.path:
    sys.path.insert(0, str(GENMO_ROOT))

from gem.datasets.pure_motion.imu_utils import (
    DEFAULT_SENSOR_COMBOS,
    build_f_imu_selected,
)
from scripts.eval.eval_imu_streaming import (
    _init_streaming_runtime,
    _make_causal_chunk_batch_reuse,
    _rollout_decoded_frames_gpu,
)


def load_genmo_model(
    ckpt_path: str,
    exp_name: str = "gem_imu_lw_rp_h_causal",
    device: torch.device | None = None,
):
    device = device or torch.device("cuda")
    config_dir = GENMO_ROOT / "configs"
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = hydra.compose(config_name="train", overrides=[f"exp={exp_name}", "use_wandb=false"])
    model = instantiate(cfg.model, _recursive_=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.pipeline.args.use_cfg_sampler_for_gen = False
    model.pipeline.denoiser3d.args.use_cfg_sampler_for_gen = False
    model = model.to(device).eval()
    return model


class GenMoLiveWrapper:
    def __init__(self, model, combo_name: str = "lw_rp_h"):
        self.model = model
        self.device = next(model.parameters()).device
        self.combo_name = combo_name
        self.history_frames = int(getattr(model, "streaming_history_frames", 96))
        self.chunk_size = int(getattr(model, "streaming_online_chunk_frames", 4))
        self.max_len = self.history_frames + self.chunk_size
        self.motion_dim = self.model.endecoder.get_motion_dim()
        self.sensor_ids = list(DEFAULT_SENSOR_COMBOS[combo_name])
        self.identity_pose = torch.eye(3, device=self.device).view(1, 1, 3, 3).repeat(1, 24, 1, 1)[0]
        self.zero_tran = torch.zeros(3, device=self.device)
        self.reset()

    def reset(self):
        self.runtime = None
        self.f_imu_frames: list[torch.Tensor] = []
        self.latent_history: list[torch.Tensor] = []
        self.rollout_state = None
        self.output_queue: deque[tuple[torch.Tensor, torch.Tensor]] = deque()
        self.last_pose = self.identity_pose.clone()
        self.last_tran = self.zero_tran.clone()

    def _build_f_imu_frame(self, acc7: torch.Tensor, ori7: torch.Tensor) -> torch.Tensor:
        acc = acc7.detach().float().cpu().clone()
        ori = ori7.detach().float().cpu().clone()
        f_imu, _, _, _ = build_f_imu_selected(
            acc.unsqueeze(0),
            ori.unsqueeze(0),
            self.sensor_ids,
            rotation_rep="mat9",
        )
        return f_imu[0]

    def _ensure_runtime(self, f_imu_dim: int):
        if self.runtime is None:
            self.runtime = _init_streaming_runtime(self.model, self.max_len, f_imu_dim, self.device)

    @torch.no_grad()
    def _run_available_chunk(self):
        processed = len(self.latent_history)
        available = len(self.f_imu_frames) - processed
        if available < self.chunk_size:
            return

        future_len = self.chunk_size
        history_len = min(processed, self.history_frames)
        left = processed - history_len
        right = processed + future_len
        f_imu_chunk = torch.stack(self.f_imu_frames[left:right]).to(self.device)
        self._ensure_runtime(f_imu_chunk.shape[-1])

        history_latents = None
        if history_len > 0:
            history_latents = torch.stack(self.latent_history[-history_len:], dim=0)

        batch = _make_causal_chunk_batch_reuse(
            self.model,
            self.runtime,
            f_imu_chunk,
            history_len,
            history_latents,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = self.model.pipeline.forward(
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
            self.latent_history.append(pred_x[i].detach())

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

        pred_pose: list[torch.Tensor] = []
        pred_tran: list[torch.Tensor] = []
        self.rollout_state = _rollout_decoded_frames_gpu(
            self.model,
            decoded,
            future_start,
            future_end,
            self.rollout_state,
            pred_pose,
            pred_tran,
        )
        for pose, tran in zip(pred_pose, pred_tran):
            self.output_queue.append((pose.detach().clone(), tran.detach().clone()))

    @torch.no_grad()
    def forward_frame(self, acc7: torch.Tensor, ori7: torch.Tensor):
        self.f_imu_frames.append(self._build_f_imu_frame(acc7, ori7))
        self._run_available_chunk()
        if self.output_queue:
            self.last_pose, self.last_tran = self.output_queue.popleft()
        return self.last_pose, self.last_tran
