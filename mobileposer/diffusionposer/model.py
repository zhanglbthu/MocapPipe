import math
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import train_hypers


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
    """Minimal DDPM training module for DiffusionPoser-style full-state modeling."""

    def __init__(self, config: DiffusionPoserConfig = None):
        super().__init__()
        self.config = config or DiffusionPoserConfig()
        self.hypers = train_hypers
        self.denoiser = DiffusionTransformer(self.config)
        self.loss = nn.MSELoss()
        self.training_step_loss = []
        self.validation_step_loss = []

        betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
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

    def shared_step(self, batch):
        x0 = batch['x0']
        lengths = batch.get('x0_lengths')
        padding_mask = None
        if lengths is not None:
            padding_mask = self._padding_mask(lengths, x0.shape[1], x0.device)

        timesteps = torch.randint(0, self.config.diffusion_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, timesteps, noise=noise)
        pred_x0 = self(x_t, timesteps, padding_mask=padding_mask)

        if padding_mask is None:
            return self.loss(pred_x0, x0)

        valid = (~padding_mask).unsqueeze(-1)
        return F.mse_loss(pred_x0[valid.expand_as(pred_x0)], x0[valid.expand_as(x0)])

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}

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
