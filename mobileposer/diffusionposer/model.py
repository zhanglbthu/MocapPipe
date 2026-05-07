import math
from dataclasses import dataclass

import articulate as art
import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import amass, datasets, paths, train_hypers


@dataclass
class DiffusionPoserConfig:
    state_dim: int = 171
    window_length: int = 125
    diffusion_steps: int = 1000
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    loss_simple_weight: float = 1.0
    loss_vel_weight: float = 1.0
    loss_fk_weight: float = 1.0
    loss_drift_weight: float = 1.0
    loss_slide_weight: float = 1.0


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        emb = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionTransformer(nn.Module):
    """Transformer denoiser that predicts clean motion state x0."""

    def __init__(self, config: DiffusionPoserConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.state_dim, config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.state_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, config.window_length, config.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, x_t, timesteps, padding_mask=None):
        seq_len = x_t.shape[1]
        if seq_len > self.pos_embed.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds configured window length {self.pos_embed.shape[1]}.")

        h = self.input_proj(x_t)
        h = h + self.pos_embed[:, :seq_len]
        h = h + self.time_embed(timesteps).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.output_proj(self.norm(h))


class DiffusionPoser(L.LightningModule):
    """DDPM training module with DiffusionPoser-style auxiliary losses."""

    pose_dim = 24 * 6
    acc_dim = 7 * 3
    root_vel_dim = 3
    root_y_dim = 1
    contact_dim = 2

    pose_slice = slice(0, pose_dim)
    acc_slice = slice(pose_dim, pose_dim + acc_dim)
    root_vel_slice = slice(acc_slice.stop, acc_slice.stop + root_vel_dim)
    root_y_slice = slice(root_vel_slice.stop, root_vel_slice.stop + root_y_dim)
    contact_slice = slice(root_y_slice.stop, root_y_slice.stop + contact_dim)

    foot_joint_ids = (10, 11)

    def __init__(self, config: DiffusionPoserConfig = None):
        super().__init__()
        self.config = config or DiffusionPoserConfig()
        self.hypers = train_hypers
        self.denoiser = DiffusionTransformer(self.config)
        self.training_step_loss = []
        self.validation_step_loss = []
        self._bodymodels = {}

        betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('norm_mean', torch.zeros(self.config.state_dim))
        self.register_buffer('norm_std', torch.ones(self.config.state_dim))
        self.save_hyperparameters()

    def q_sample(self, x0, timesteps, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        x_t = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
        return x_t, noise

    def forward(self, x_t, timesteps, padding_mask=None):
        return self.denoiser(x_t, timesteps, padding_mask=padding_mask)

    def _padding_mask(self, lengths, max_len, device):
        lengths = torch.tensor(lengths, device=device)
        frame_ids = torch.arange(max_len, device=device).unsqueeze(0)
        return frame_ids >= lengths.unsqueeze(1)

    def _bodymodel(self, device):
        key = str(device)
        if key not in self._bodymodels:
            self._bodymodels[key] = art.model.ParametricModel(paths.smpl_file, device=device)
        return self._bodymodels[key]

    def set_normalization_stats(self, mean, std):
        mean = mean.to(device=self.norm_mean.device, dtype=self.norm_mean.dtype)
        std = std.to(device=self.norm_std.device, dtype=self.norm_std.dtype).clamp_min(1e-6)
        if mean.shape != self.norm_mean.shape or std.shape != self.norm_std.shape:
            raise ValueError(
                f"Normalization stats shape mismatch: expected {self.norm_mean.shape}, "
                f"got mean={mean.shape}, std={std.shape}."
            )
        self.norm_mean.copy_(mean)
        self.norm_std.copy_(std)

    def normalize_state(self, state):
        return (state - self.norm_mean) / self.norm_std

    def denormalize_state(self, state):
        return state * self.norm_std + self.norm_mean

    def _masked_mse(self, pred, target, mask=None):
        sq = (pred - target) ** 2
        if mask is None:
            return sq.mean()

        mask = mask.to(dtype=sq.dtype)
        while mask.dim() < sq.dim():
            mask = mask.unsqueeze(-1)
        expanded_mask = mask.expand_as(sq)
        denom = expanded_mask.sum().clamp_min(1.0)
        return (sq * expanded_mask).sum() / denom

    def _state_to_pose_matrix(self, state):
        pose_6d = state[..., self.pose_slice].contiguous().view(*state.shape[:-1], 24, 6)
        pose_mat = art.math.r6d_to_rotation_matrix(pose_6d)
        return pose_mat.view(*state.shape[:-1], 24, 3, 3)

    def _state_to_root_delta(self, state):
        root_delta = state[..., self.root_vel_slice] / (datasets.fps / amass.vel_scale)
        return root_delta

    def _cumulative_root_delta(self, root_delta):
        accum = torch.zeros_like(root_delta[:, :1])
        pieces = []
        for t in range(root_delta.shape[1]):
            accum = accum + root_delta[:, t:t + 1]
            pieces.append(accum)
        return torch.cat(pieces, dim=1)

    def _state_to_tran(self, state):
        root_delta = self._state_to_root_delta(state)
        tran = self._cumulative_root_delta(root_delta)
        tran[..., 1:2] = state[..., self.root_y_slice]
        return tran

    def _global_pose_to_joint(self, pose_global):
        bodymodel = self._bodymodel(pose_global.device)
        bsz, seqlen = pose_global.shape[:2]
        pose_local = bodymodel.inverse_kinematics_R(pose_global.view(-1, 24, 3, 3))
        _, joints = bodymodel.forward_kinematics(pose_local.view(-1, 24, 3, 3))
        return joints.view(bsz, seqlen, 24, 3)

    def _compute_losses(self, pred_x0, batch, valid_mask=None):
        target_x0 = batch["x0"]
        target_x0_norm = self.normalize_state(target_x0)
        pred_x0_raw = self.denormalize_state(pred_x0)

        pose_p = self._state_to_pose_matrix(pred_x0_raw)
        pose_t = self._state_to_pose_matrix(target_x0)

        simple = self._masked_mse(pred_x0, target_x0_norm, valid_mask)

        if pose_p.shape[1] > 1:
            pair_mask = valid_mask[:, 1:] * valid_mask[:, :-1] if valid_mask is not None else None
            vel_p = pose_p[:, 1:] - pose_p[:, :-1]
            vel_t = pose_t[:, 1:] - pose_t[:, :-1]
            vel = self._masked_mse(vel_p.flatten(2), vel_t.flatten(2), pair_mask)
        else:
            vel = pred_x0.new_zeros(())

        joint_p = self._global_pose_to_joint(pose_p)
        joint_t = batch["joint"]
        fk = self._masked_mse(joint_p.flatten(2), joint_t.flatten(2), valid_mask)

        tran_p = self._state_to_tran(pred_x0_raw)
        tran_t = batch["tran"]
        drift = self._masked_mse(tran_p, tran_t, valid_mask)

        if pose_p.shape[1] > 1:
            pair_mask = valid_mask[:, 1:] * valid_mask[:, :-1] if valid_mask is not None else None
            foot_p = joint_p[:, :, self.foot_joint_ids]
            foot_delta = foot_p[:, 1:] - foot_p[:, :-1]
            root_delta = self._state_to_root_delta(pred_x0_raw)[:, 1:].clone()
            root_delta[..., 1] = 0.0
            contact = pred_x0_raw[:, :-1, self.contact_slice].sigmoid()
            slide_target = foot_delta + root_delta.unsqueeze(2)
            slide = self._masked_mse(contact.unsqueeze(-1) * slide_target, torch.zeros_like(slide_target), pair_mask)
        else:
            slide = pred_x0.new_zeros(())

        total = (
            self.config.loss_simple_weight * simple
            + self.config.loss_vel_weight * vel
            + self.config.loss_fk_weight * fk
            + self.config.loss_drift_weight * drift
            + self.config.loss_slide_weight * slide
        )

        return {
            "loss": total,
            "loss_simple": simple,
            "loss_vel": vel,
            "loss_fk": fk,
            "loss_drift": drift,
            "loss_slide": slide,
        }

    def shared_step(self, batch):
        x0 = self.normalize_state(batch['x0'])
        lengths = batch.get('x0_lengths')
        padding_mask = None
        valid_mask = None
        if lengths is not None:
            padding_mask = self._padding_mask(lengths, x0.shape[1], x0.device)
            valid_mask = ~padding_mask

        timesteps = torch.randint(0, self.config.diffusion_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, timesteps, noise=noise)
        pred_x0 = self(x_t, timesteps, padding_mask=padding_mask)
        return self._compute_losses(pred_x0, batch, valid_mask=valid_mask)

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        self.log("training_step_loss", losses["loss"].item(), batch_size=self.hypers.batch_size)
        self.log("training_step_loss_simple", losses["loss_simple"].item(), batch_size=self.hypers.batch_size)
        self.log("training_step_loss_vel", losses["loss_vel"].item(), batch_size=self.hypers.batch_size)
        self.log("training_step_loss_fk", losses["loss_fk"].item(), batch_size=self.hypers.batch_size)
        self.log("training_step_loss_drift", losses["loss_drift"].item(), batch_size=self.hypers.batch_size)
        self.log("training_step_loss_slide", losses["loss_slide"].item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(losses["loss"].item())
        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        self.log("validation_step_loss", losses["loss"].item(), batch_size=self.hypers.batch_size)
        self.log("validation_step_loss_simple", losses["loss_simple"].item(), batch_size=self.hypers.batch_size)
        self.log("validation_step_loss_vel", losses["loss_vel"].item(), batch_size=self.hypers.batch_size)
        self.log("validation_step_loss_fk", losses["loss_fk"].item(), batch_size=self.hypers.batch_size)
        self.log("validation_step_loss_drift", losses["loss_drift"].item(), batch_size=self.hypers.batch_size)
        self.log("validation_step_loss_slide", losses["loss_slide"].item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(losses["loss"].item())
        return {"loss": losses["loss"]}

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()

    def epoch_end_callback(self, outputs, loop_type):
        if not outputs:
            return
        average_loss = torch.mean(torch.tensor(outputs, device=self.device))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hypers.lr)
