from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import articulate as art


def _rotation_diversity(rot: torch.Tensor):
    """
    rot: [1, T, D, 3, 3]
    returns: [D] diversity count
    """
    n_batch, seq_len, imu_num = rot.shape[:3]
    rot = rot.reshape(-1, 3, 3)
    euler = art.math.rotation_matrix_to_euler_angle(rot).reshape(n_batch, seq_len, imu_num, 3) * 180 / torch.pi
    dis_angle = torch.div(euler, 15, rounding_mode="floor").long()
    dis_angle = dis_angle + torch.tensor([12, 6, 12], device=euler.device).view(1, 1, 1, 3)
    dis_angle_idx = (
        torch.clamp(dis_angle[:, :, :, [0]], 0, 23)
        + torch.clamp(dis_angle[:, :, :, [1]], 0, 11) * 24
        + torch.clamp(dis_angle[:, :, :, [2]], 0, 23) * 12 * 24
    )
    angle_space = torch.zeros(n_batch, seq_len, imu_num, 24 * 12 * 24, dtype=torch.uint8, device=euler.device)
    angle_space.scatter_add_(3, dis_angle_idx, torch.ones_like(angle_space, dtype=torch.uint8))
    angle_space_sum = angle_space.sum(dim=1)
    angle_space_mask = (angle_space_sum > 0).reshape(n_batch, imu_num, -1)
    diversity = angle_space_mask.sum(dim=-1)
    return diversity[0]


def ego_drift_regularization(rot: torch.Tensor, ego_idx: int = -1):
    """
    rot: [D, 3, 3]
    Zero out ego yaw drift, following TIC.
    """
    rot = rot.clone()
    rot_ego = rot[ego_idx]
    rot_ego_euler = art.math.rotation_matrix_to_euler_angle(rot_ego, seq="YZX").squeeze(0)
    rot_ego_euler[0] *= 0
    rot[ego_idx] = art.math.euler_angle_to_rotation_matrix(rot_ego_euler, seq="YZX")
    return rot


@torch.no_grad()
def simulate_imu_drift_offset(
    imu_rot: torch.Tensor,
    imu_acc: torch.Tensor,
    imu_num: int,
    ego_imu_id: int = -1,
    drift_range: float = 60.0,
    offset_range: float = 45.0,
    random_global_yaw: bool = True,
    global_yaw_only: bool = False,
    acc_noise: float = 0.025,
):
    """
    Adapted from TIC's simulations.py.

    imu_rot: [B, T, D, 3, 3]
    imu_acc: [B, T, D, 3]
    returns:
      noisy_rot [B, T, D, 3, 3]
      noisy_acc [B, T, D, 3]
      drift_r6d [B, D, 6]
      offset_r6d [B, D, 6]
    """
    batch_size, seq_len = imu_rot.shape[:2]
    drift_range = (drift_range / 180.0) * torch.pi
    non_yaw_drift_range = (20.0 / 180.0) * torch.pi
    offset_range = (offset_range / 180.0) * torch.pi
    ga = torch.tensor([[0.0, -9.80665, 0.0]], device=imu_rot.device)

    imu_acc = imu_acc + torch.randn_like(imu_acc) * acc_noise

    if global_yaw_only:
        drift = torch.eye(3, device=imu_rot.device).view(1, 1, 3, 3).repeat(batch_size, imu_num, 1, 1)
        offset = drift.clone()
    else:
        drift_euler = torch.zeros(batch_size, imu_num, 3, device=imu_rot.device)
        offset_euler = torch.zeros(batch_size, imu_num, 3, device=imu_rot.device)
        drift_euler[:, :, 0].uniform_(-drift_range, drift_range)
        drift_euler[:, :, 1:].uniform_(-non_yaw_drift_range, non_yaw_drift_range)
        drift_euler[:, ego_imu_id, 0] *= 0
        offset_euler.uniform_(-offset_range, offset_range)

        scale_mask = torch.zeros(batch_size, 1, 1, device=imu_rot.device).uniform_(0, 1)
        drift_euler *= scale_mask
        offset_euler *= scale_mask

        drift = art.math.euler_angle_to_rotation_matrix(drift_euler, seq="YZX").reshape(batch_size, imu_num, 3, 3)
        offset = art.math.euler_angle_to_rotation_matrix(offset_euler).reshape(batch_size, imu_num, 3, 3)

    if random_global_yaw:
        global_yaw_rot = torch.zeros(batch_size, 1, 3, device=imu_rot.device)
        global_yaw_rot[:, :, 1].uniform_(-torch.pi, torch.pi)
        global_yaw_rot = art.math.euler_angle_to_rotation_matrix(global_yaw_rot).reshape(batch_size, 1, 3, 3)
        global_yaw_rot = global_yaw_rot.unsqueeze(1).repeat(1, seq_len, imu_num, 1, 1)
        imu_rot = global_yaw_rot.matmul(imu_rot)
        imu_acc = global_yaw_rot.matmul(imu_acc.unsqueeze(-1)).squeeze(-1)

    if not global_yaw_only:
        drift_seq = drift.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        offset_seq = offset.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        imu_rot = drift_seq.matmul(imu_rot).matmul(offset_seq)

        ga = ga.view(1, 1, 1, 3, 1).repeat(batch_size, seq_len, imu_num, 1, 1)
        imu_acc = drift_seq.matmul(imu_acc.unsqueeze(-1)) + (
            torch.eye(3, device=imu_acc.device).view(1, 1, 1, 3, 3) - drift_seq
        ).matmul(ga)
        imu_acc = imu_acc.squeeze(-1)

    drift_r6d = art.math.rotation_matrix_to_r6d(drift.reshape(-1, 3, 3)).reshape(batch_size, imu_num, 6)
    offset_r6d = art.math.rotation_matrix_to_r6d(offset.reshape(-1, 3, 3)).reshape(batch_size, imu_num, 6)
    return imu_rot, imu_acc, drift_r6d, offset_r6d


class TICHead(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, n_output: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.mapping = nn.Linear(d_model, n_output)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is None:
            x = x.mean(dim=1)
        else:
            valid = (~src_key_padding_mask).unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1)
            x = (x * valid).sum(dim=1) / denom
        return self.mapping(x)


class TICTransformerCalibrator(nn.Module):
    """
    Faithful TIC-style transformer:
    sequence in -> predict per-device drift and offset as 6D rotations.
    """

    def __init__(
        self,
        imu_num: int,
        n_input: int,
        stack: int = 4,
        multi_head: int = 8,
        d_model: int = 256,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.imu_num = imu_num
        self.n_input = n_input
        self.n_output = imu_num * 6
        self.input_embedding_layer = nn.Linear(n_input, d_model)
        encoder_layers = [
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=multi_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(stack)
        ]
        self.encoder_backbone = nn.ModuleList(encoder_layers)
        self.tpm_global = TICHead(nhead=multi_head, d_model=d_model, d_ff=d_ff, n_output=self.n_output, dropout=dropout)
        self.tpm_local = TICHead(nhead=multi_head, d_model=d_model, d_ff=d_ff, n_output=self.n_output, dropout=dropout)

    def forward(self, x: torch.Tensor, seq_mask: Optional[torch.Tensor] = None):
        padding_mask = None if seq_mask is None else ~seq_mask
        x = self.input_embedding_layer(x)
        for encoder_layer in self.encoder_backbone:
            x = encoder_layer(x, src_key_padding_mask=padding_mask)
        global_shift = self.tpm_global(x, src_key_padding_mask=padding_mask).view(-1, self.imu_num, 6)
        local_shift = self.tpm_local(x, src_key_padding_mask=padding_mask).view(-1, self.imu_num, 6)
        return global_shift, local_shift


def apply_calibration(
    acc: torch.Tensor,
    rot: torch.Tensor,
    r_dg: torch.Tensor,
    r_bs: torch.Tensor,
    gravity: Optional[torch.Tensor] = None,
):
    """
    acc: [..., D, 3]
    rot: [..., D, 3, 3]
    r_dg/r_bs: [D, 3, 3]
    """
    if gravity is None:
        gravity = torch.tensor([0.0, -9.80848, 0.0], device=acc.device)
    gravity = gravity.view(*([1] * (acc.dim() - 1)), 3, 1).expand(*acc.shape[:-1], 3, 1)
    acc_col = acc.unsqueeze(-1)
    rot = r_dg.transpose(-2, -1).matmul(rot).matmul(r_bs.transpose(-2, -1))
    acc = r_dg.transpose(-2, -1).matmul(acc_col - gravity) + gravity
    return acc.squeeze(-1), rot


@dataclass
class TICOperatorConfig:
    buffer_size: int = 512
    trigger_t: float = 1.0
    data_frame_rate: int = 30
    drift_threshold: Optional[list] = None
    offset_threshold: Optional[list] = None
    ego_idx: int = -1


class TICOnlineCalibrator:
    """
    Sequence operator adapted from TIC's TicOperator.
    """

    def __init__(self, model: TICTransformerCalibrator, imu_num: int, config: TICOperatorConfig):
        self.model = model
        self.imu_num = imu_num
        self.config = config
        self.device = next(model.parameters()).device
        if config.drift_threshold is None:
            config.drift_threshold = [10.0] * max(imu_num - 1, 0) + [0.0]
        if config.offset_threshold is None:
            config.offset_threshold = [30.0] * imu_num
        self.reset()

    def reset(self):
        self.r_dg = torch.eye(3).reshape(1, 3, 3).repeat(self.imu_num, 1, 1)
        self.r_bs = torch.eye(3).reshape(1, 3, 3).repeat(self.imu_num, 1, 1)
        self.ga = torch.tensor([[0.0, -9.80848, 0.0]]).repeat(self.imu_num, 1).unsqueeze(-1)
        self.data_buffer = []
        self.frame_idx = 0

    @torch.no_grad()
    def calibrate_step(self, acc_cat_rot: torch.Tensor):
        acc = acc_cat_rot[: self.imu_num * 3].view(self.imu_num, 3, 1)
        rot = acc_cat_rot[self.imu_num * 3 :].view(self.imu_num, 3, 3)
        rot = self.r_dg.transpose(-2, -1).matmul(rot).matmul(self.r_bs.transpose(-2, -1))
        acc = self.r_dg.transpose(-2, -1).matmul(acc - self.ga) + self.ga
        return torch.cat([acc.flatten(), rot.flatten()], dim=0)

    @torch.no_grad()
    def dynamic_calibration(self):
        if len(self.data_buffer) < self.config.buffer_size:
            return
        stride = max(self.config.data_frame_rate // 30, 1)
        acc_cat_rot = torch.stack(self.data_buffer[-self.config.buffer_size :])[::stride]
        acc_cat_rot[:, : self.imu_num * 3] /= 30.0
        acc_cat_rot = acc_cat_rot.to(self.device)

        self.r_dg = art.math.r6d_to_rotation_matrix(art.math.rotation_matrix_to_r6d(self.r_dg))
        self.r_bs = art.math.r6d_to_rotation_matrix(art.math.rotation_matrix_to_r6d(self.r_bs))

        oris = acc_cat_rot[:, self.imu_num * 3 :].reshape(1, -1, self.imu_num, 3, 3)
        diversity = _rotation_diversity(oris).reshape(-1)
        acc_cat_rot = acc_cat_rot.reshape(1, -1, self.imu_num * 12)
        delta_r_dg, delta_r_bs = self.model(acc_cat_rot)
        delta_r_dg = art.math.r6d_to_rotation_matrix(delta_r_dg.reshape(-1, 6)).cpu()
        delta_r_bs = art.math.r6d_to_rotation_matrix(delta_r_bs.reshape(-1, 6)).cpu()
        delta_r_dg = ego_drift_regularization(delta_r_dg, ego_idx=self.config.ego_idx)

        drift_th = torch.tensor(self.config.drift_threshold, device=diversity.device)
        offset_th = torch.tensor(self.config.offset_threshold, device=diversity.device)
        skip_mask_drift = ~(diversity > drift_th)
        skip_mask_offset = ~(diversity > offset_th)

        if skip_mask_drift.sum().item() < self.imu_num or skip_mask_offset.sum().item() < self.imu_num:
            if skip_mask_drift.any():
                delta_r_dg[skip_mask_drift] = torch.eye(3).unsqueeze(0).repeat(skip_mask_drift.sum().item(), 1, 1)
            if skip_mask_offset.any():
                delta_r_bs[skip_mask_offset] = torch.eye(3).unsqueeze(0).repeat(skip_mask_offset.sum().item(), 1, 1)
            self.r_dg = self.r_dg.matmul(delta_r_dg)
            self.r_bs = delta_r_bs.matmul(self.r_bs)
            self.data_buffer = []

    @torch.no_grad()
    def forward_frame(self, rot: torch.Tensor, acc: torch.Tensor):
        """
        rot: [D, 3, 3]
        acc: [D, 3]
        returns:
          calibrated_rot: [D, 3, 3]
          calibrated_acc: [D, 3]
        """
        acc_cat_rot = torch.cat([acc.reshape(-1), rot.reshape(-1)], dim=0).cpu()
        recali = self.calibrate_step(acc_cat_rot)
        self.data_buffer.append(recali.clone())

        trigger_gap = int(self.config.data_frame_rate * self.config.trigger_t)
        if self.frame_idx % trigger_gap == 0:
            self.dynamic_calibration()
        if len(self.data_buffer) > self.config.buffer_size:
            self.data_buffer = self.data_buffer[-self.config.buffer_size :]
        self.frame_idx += 1

        calibrated_acc = recali[: self.imu_num * 3].reshape(self.imu_num, 3)
        calibrated_rot = recali[self.imu_num * 3 :].reshape(self.imu_num, 3, 3)
        return calibrated_rot, calibrated_acc

    @torch.no_grad()
    def run(self, rot: torch.Tensor, acc: torch.Tensor):
        self.reset()
        trigger_gap = int(self.config.data_frame_rate * self.config.trigger_t)
        acc = acc.reshape(-1, self.imu_num * 3)
        rot = rot.reshape(-1, self.imu_num * 9)
        origin = torch.cat([acc, rot], dim=-1)
        recali = []
        pred_drift = []
        pred_offset = []
        for i in range(len(origin)):
            recali.append(self.calibrate_step(origin[i]))
            self.data_buffer.append(recali[-1].clone())
            pred_drift.append(self.r_dg.clone())
            pred_offset.append(self.r_bs.clone())
            if i % trigger_gap == 0:
                self.dynamic_calibration()
            if len(self.data_buffer) > self.config.buffer_size:
                self.data_buffer = self.data_buffer[-self.config.buffer_size :]
        recali = torch.stack(recali, dim=0)
        acc = recali[:, : self.imu_num * 3].reshape(-1, self.imu_num, 3)
        rot = recali[:, self.imu_num * 3 :].reshape(-1, self.imu_num, 3, 3)
        return rot, acc, torch.stack(pred_drift), torch.stack(pred_offset)
