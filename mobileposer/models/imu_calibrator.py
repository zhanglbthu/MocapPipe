import torch
import torch.nn as nn
from typing import Optional

import articulate as art

try:
    from mambapy.mamba import Mamba as StandardMamba
    from mambapy.mamba import MambaConfig as StandardMambaConfig
except ImportError:
    StandardMamba = None
    StandardMambaConfig = None


def build_imu_input(acc: torch.Tensor, ori: torch.Tensor):
    ori9d = ori.reshape(ori.shape[0], ori.shape[1], 9)
    return torch.cat([acc, ori9d], dim=-1)


def extract_input_ori(x: torch.Tensor, combo_size: int, input_dim_per_device: int):
    if input_dim_per_device != 12:
        raise ValueError("Residual orientation calibration expects per-device features [acc(3), ori(9)].")
    if x.dim() == 4:
        return x[..., 3:].reshape(*x.shape[:-1], 3, 3)
    if x.dim() == 3:
        bsz, seq_len, flat_dim = x.shape
        expected = combo_size * input_dim_per_device
        if flat_dim != expected:
            raise ValueError(f"Expected last dim {expected}, got {flat_dim}")
        return x.view(bsz, seq_len, combo_size, input_dim_per_device)[..., 3:].reshape(bsz, seq_len, combo_size, 3, 3)
    raise ValueError(f"Unsupported input shape {tuple(x.shape)}")


def compose_residual_orientation(input_ori: torch.Tensor, delta_ori6d: torch.Tensor):
    delta_ori = art.math.r6d_to_rotation_matrix(delta_ori6d.reshape(-1, 6)).view(*delta_ori6d.shape[:-1], 3, 3)
    return torch.matmul(input_ori, delta_ori), delta_ori


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return x + self.net(x)


class CrossDeviceAttentionBlock(nn.Module):
    """
    Fuse synchronous observations across devices at each timestep.
    Input/output: [B, T, D, H]
    """

    def __init__(self, hidden_dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForwardBlock(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, num_devices, hidden_dim = x.shape
        flat = x.reshape(bsz * seq_len, num_devices, hidden_dim)
        flat_norm = self.norm(flat)
        attn_out, _ = self.attn(flat_norm, flat_norm, flat_norm, need_weights=False)
        flat = flat + attn_out
        flat = self.ff(flat)
        return flat.reshape(bsz, seq_len, num_devices, hidden_dim)


class ConfidenceGatedCrossDeviceFusion(nn.Module):
    """
    Explicit cooperative fusion between device-local features and
    cross-device aggregated features.
    Input/output: [B, T, D, H]
    """

    def __init__(self, hidden_dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_block = CrossDeviceAttentionBlock(hidden_dim=hidden_dim, nhead=nhead, dropout=dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        fused = self.attn_block(x)
        gate = torch.sigmoid(self.gate(torch.cat([x, fused], dim=-1)))
        return (1.0 - gate) * x + gate * fused, gate


class SelectiveStateSpaceBlock(nn.Module):
    """
    Lightweight state-space style block used as a fallback when mamba_ssm
    is unavailable. It performs a gated diagonal recurrent scan.
    Input/output: [N, T, H]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.x_proj = nn.Linear(hidden_dim, hidden_dim)
        self.g_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.a_log = nn.Parameter(torch.zeros(hidden_dim))

    def _scan(self, x: torch.Tensor):
        x_norm = self.norm(x)
        x_in = torch.tanh(self.x_proj(x_norm))
        gate = torch.sigmoid(self.g_proj(x_norm))
        dt = torch.nn.functional.softplus(self.dt_proj(x_norm))
        a = -torch.exp(self.a_log).view(1, 1, -1)
        state = torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            decay = torch.exp(a[:, 0] * dt[:, t])
            state = decay * state + (1.0 - decay) * x_in[:, t]
            outputs.append(state * gate[:, t])
        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)
        y = self.dropout(y)
        return x + y

    def forward(self, x: torch.Tensor, bidirectional: bool = False):
        if not bidirectional:
            return self._scan(x)
        y_fwd = self._scan(x)
        y_bwd = torch.flip(self._scan(torch.flip(x, dims=[1])), dims=[1])
        return 0.5 * (y_fwd + y_bwd)


class StandardMambaSequenceBlock(nn.Module):
    """
    Standard Mamba sequence block backed by `mambapy`.
    This is used for the backbone-replacement baseline to avoid relying on the
    custom fallback state-space recurrence.
    """

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        if StandardMamba is None or StandardMambaConfig is None:
            raise ImportError("mambapy is required for StandardMambaSequenceBlock")
        del dropout  # mambapy config does not expose a direct dropout knob.
        self.config = StandardMambaConfig(
                d_model=hidden_dim,
                n_layers=num_layers,
                use_cuda=False,
            )
        self.mamba = StandardMamba(self.config)
        self.norm = nn.LayerNorm(hidden_dim)
        self.caches = None

    def forward(self, x: torch.Tensor, bidirectional: bool = False):
        if not bidirectional:
            return self.norm(self.mamba(x))
        y_fwd = self.mamba(x)
        y_bwd = torch.flip(self.mamba(torch.flip(x, dims=[1])), dims=[1])
        return self.norm(0.5 * (y_fwd + y_bwd))

    def reset_cache(self):
        self.caches = None

    def _init_caches(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        cfg = self.config
        h = torch.zeros(batch_size, cfg.d_inner, cfg.d_state, device=device, dtype=dtype)
        inputs = torch.zeros(batch_size, cfg.d_inner, cfg.d_conv - 1, device=device, dtype=dtype)
        self.caches = [(h.clone(), inputs.clone()) for _ in range(cfg.n_layers)]

    def step(self, x: torch.Tensor):
        if self.caches is None:
            self._init_caches(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        y, self.caches = self.mamba.step(x, self.caches)
        return self.norm(y)


class CrossDeviceMambaCalibrator(nn.Module):
    """
    Cross-device calibrator with two explicit stages:
    1) per-timestep fusion across IMU devices
    2) temporal state-space modeling per device

    This version combines explicit cross-device fusion with a standard
    Mamba temporal backbone. During offline training it supports non-causal
    bidirectional sequence modeling; during online inference it maintains
    Mamba state and outputs framewise calibrated orientations directly.
    """

    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        predict_acc: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 256,
        online_past_frames: int = 45,
        online_future_frames: int = 5,
    ):
        super().__init__()
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.input_dim = combo_size * input_dim_per_device
        self.predict_acc = predict_acc
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.online_past_frames = online_past_frames
        self.online_future_frames = online_future_frames

        self.input_proj = nn.Linear(input_dim_per_device, hidden_dim)
        self.device_embed = nn.Embedding(combo_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, hidden_dim))
        self.fusion_blocks = nn.ModuleList(
            [CrossDeviceAttentionBlock(hidden_dim=hidden_dim, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.temporal_blocks = nn.ModuleList(
            [StandardMambaSequenceBlock(hidden_dim=hidden_dim, num_layers=1, dropout=dropout) for _ in range(num_layers)]
        )
        self.device_norm = nn.LayerNorm(hidden_dim)
        self.acc_head = nn.Linear(hidden_dim, 3) if predict_acc else None
        self.ori_head = nn.Linear(hidden_dim, 6)
        self.buffer = None
        self.stream_step = 0

    def reset(self):
        self.buffer = None
        self.stream_step = 0
        for block in self.temporal_blocks:
            block.reset_cache()

    def _prepare_input(self, x: torch.Tensor):
        if x.dim() == 4:
            batch_size, seq_len, combo_size, feat_dim = x.shape
            if combo_size != self.combo_size or feat_dim != self.input_dim_per_device:
                raise ValueError(
                    f"Expected [B,T,{self.combo_size},{self.input_dim_per_device}], got {tuple(x.shape)}"
                )
            return x
        if x.dim() == 3:
            batch_size, seq_len, flat_dim = x.shape
            if flat_dim != self.input_dim:
                raise ValueError(f"Expected last dim {self.input_dim}, got {flat_dim}")
            return x.view(batch_size, seq_len, self.combo_size, self.input_dim_per_device)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        return_ori6d: bool = False,
    ):
        x = self._prepare_input(x)
        batch_size, seq_len, combo_size, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        feat = self.input_proj(x)
        device_ids = torch.arange(combo_size, device=feat.device)
        feat = feat + self.device_embed(device_ids).view(1, 1, combo_size, self.hidden_dim)
        feat = feat + self.pos_embed[:, :seq_len]

        for fusion_block, temporal_block in zip(self.fusion_blocks, self.temporal_blocks):
            feat = fusion_block(feat)
            temp = feat.permute(0, 2, 1, 3).reshape(batch_size * combo_size, seq_len, self.hidden_dim)
            temp = temporal_block(temp, bidirectional=not causal)
            feat = temp.view(batch_size, combo_size, seq_len, self.hidden_dim).permute(0, 2, 1, 3)

        feat = self.device_norm(feat)

        input_ori = x[..., 3:].reshape(batch_size, seq_len, combo_size, 3, 3)
        pred_acc = None
        if self.acc_head is not None:
            pred_acc = self.acc_head(feat)
        pred_ori6d = self.ori_head(feat)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)

        if seq_mask is not None:
            mask = seq_mask.view(batch_size, seq_len, 1, 1)
            if pred_acc is not None:
                pred_acc = pred_acc * mask
            pred_ori6d = pred_ori6d * mask

        if return_ori6d:
            return pred_acc, pred_ori, pred_ori6d
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame_windowed(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        if self.buffer is None:
            recent = frame.unsqueeze(0).repeat(self.online_past_frames, 1)
        else:
            recent = torch.cat([self.buffer[1:], frame.unsqueeze(0)], dim=0)

        future = frame.unsqueeze(0).repeat(self.online_future_frames, 1)
        window = torch.cat([recent, future], dim=0)
        seq = window.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=False)
        self.buffer = recent

        output_idx = self.online_past_frames - 1
        pred_acc_frame = None if pred_acc is None else pred_acc[0, output_idx]
        return pred_acc_frame, pred_ori[0, output_idx]

    @torch.no_grad()
    def forward_frame(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        x = frame.view(1, self.combo_size, self.input_dim_per_device)
        feat = self.input_proj(x)
        device_ids = torch.arange(self.combo_size, device=feat.device)
        feat = feat + self.device_embed(device_ids).view(1, self.combo_size, self.hidden_dim)
        pos_idx = min(self.stream_step, self.max_seq_len - 1)
        feat = feat + self.pos_embed[:, pos_idx]
        feat = feat.unsqueeze(1)

        for fusion_block, temporal_block in zip(self.fusion_blocks, self.temporal_blocks):
            feat = fusion_block(feat)
            temp = feat[:, 0]
            temp = temporal_block.step(temp.reshape(self.combo_size, self.hidden_dim))
            feat = temp.view(1, 1, self.combo_size, self.hidden_dim)

        feat = self.device_norm(feat[:, 0])
        pred_acc_frame = None
        if self.acc_head is not None:
            pred_acc_frame = self.acc_head(feat).view(self.combo_size, 3)
        pred_ori6d = self.ori_head(feat).view(self.combo_size, 6)
        input_ori = x[..., 3:].reshape(self.combo_size, 3, 3)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)
        self.stream_step += 1
        return pred_acc_frame, pred_ori


class BackboneMambaCalibrator(nn.Module):
    """
    Pure backbone-replacement baseline.

    This keeps the same direct calibrated-IMU regression setup as the
    combo transformer calibrator, but swaps the temporal backbone from
    Transformer to a stack of lightweight state-space blocks. Unlike
    `CrossDeviceMambaCalibrator`, it does not add explicit cross-device
    interaction beyond flattening the combo into one frame token.
    """

    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        predict_acc: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 256,
        online_past_frames: int = 45,
        online_future_frames: int = 5,
    ):
        super().__init__()
        del nhead  # Unused here; kept for CLI compatibility with transformer models.
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.input_dim = combo_size * input_dim_per_device
        self.predict_acc = predict_acc
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.online_past_frames = online_past_frames
        self.online_future_frames = online_future_frames

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.temporal_backbone = StandardMambaSequenceBlock(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.acc_head = nn.Linear(hidden_dim, combo_size * 3) if predict_acc else None
        self.ori_head = nn.Linear(hidden_dim, combo_size * 6)
        self.buffer = None
        self.stream_step = 0

    def reset(self):
        self.buffer = None
        self.stream_step = 0
        self.temporal_backbone.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        return_ori6d: bool = False,
    ):
        if x.dim() == 4:
            batch_size, seq_len, combo_size, feat_dim = x.shape
            if combo_size != self.combo_size or feat_dim != self.input_dim_per_device:
                raise ValueError(f"Expected [B,T,{self.combo_size},{self.input_dim_per_device}], got {tuple(x.shape)}")
            flat_x = x.reshape(batch_size, seq_len, -1)
        elif x.dim() == 3:
            batch_size, seq_len, flat_dim = x.shape
            if flat_dim != self.input_dim:
                raise ValueError(f"Expected last dim {self.input_dim}, got {flat_dim}")
            flat_x = x
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        input_ori = extract_input_ori(x, self.combo_size, self.input_dim_per_device)
        feat = self.input_proj(flat_x)
        feat = feat + self.pos_embed[:, :seq_len]

        feat = self.temporal_backbone(feat, bidirectional=not causal)
        feat = self.norm(feat)

        pred_acc = None
        if self.acc_head is not None:
            pred_acc = self.acc_head(feat).view(batch_size, seq_len, self.combo_size, 3)
        pred_ori6d = self.ori_head(feat).view(batch_size, seq_len, self.combo_size, 6)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)

        if seq_mask is not None:
            mask = seq_mask.view(batch_size, seq_len, 1, 1)
            if pred_acc is not None:
                pred_acc = pred_acc * mask
            pred_ori6d = pred_ori6d * mask

        if return_ori6d:
            return pred_acc, pred_ori, pred_ori6d
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame_windowed(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        if self.buffer is None:
            recent = frame.unsqueeze(0).repeat(self.online_past_frames, 1)
        else:
            recent = torch.cat([self.buffer[1:], frame.unsqueeze(0)], dim=0)

        future = frame.unsqueeze(0).repeat(self.online_future_frames, 1)
        window = torch.cat([recent, future], dim=0)
        seq = window.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=False)
        self.buffer = recent

        output_idx = self.online_past_frames - 1
        pred_acc_frame = None if pred_acc is None else pred_acc[0, output_idx]
        return pred_acc_frame, pred_ori[0, output_idx]

    @torch.no_grad()
    def forward_frame(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        feat = self.input_proj(frame.unsqueeze(0))
        pos_idx = min(self.stream_step, self.max_seq_len - 1)
        feat = feat + self.pos_embed[:, pos_idx]
        feat = self.temporal_backbone.step(feat)
        feat = self.norm(feat)

        pred_acc_frame = None
        if self.acc_head is not None:
            pred_acc_frame = self.acc_head(feat).view(self.combo_size, 3)
        pred_ori6d = self.ori_head(feat).view(self.combo_size, 6)
        input_ori = frame.view(self.combo_size, self.input_dim_per_device)[:, 3:].reshape(self.combo_size, 3, 3)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)
        self.stream_step += 1
        return pred_acc_frame, pred_ori


class BackboneLSTMCalibrator(nn.Module):
    """
    Pure backbone-replacement baseline with a unidirectional LSTM.

    This keeps the same direct calibrated-orientation regression setup as the
    combo transformer calibrator, but swaps the temporal backbone to a
    stateful unidirectional LSTM for true streaming inference.
    """

    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        predict_acc: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 256,
        online_past_frames: int = 45,
        online_future_frames: int = 5,
    ):
        super().__init__()
        del nhead  # Unused; kept for CLI compatibility.
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.input_dim = combo_size * input_dim_per_device
        self.predict_acc = predict_acc
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.online_past_frames = online_past_frames
        self.online_future_frames = online_future_frames

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.temporal_backbone = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.acc_head = nn.Linear(hidden_dim, combo_size * 3) if predict_acc else None
        self.ori_head = nn.Linear(hidden_dim, combo_size * 6)
        self.buffer = None
        self.stream_step = 0
        self.hidden_state = None

    def reset(self):
        self.buffer = None
        self.stream_step = 0
        self.hidden_state = None

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        return_ori6d: bool = False,
    ):
        del causal  # Unidirectional LSTM is inherently causal.
        if x.dim() == 4:
            batch_size, seq_len, combo_size, feat_dim = x.shape
            if combo_size != self.combo_size or feat_dim != self.input_dim_per_device:
                raise ValueError(f"Expected [B,T,{self.combo_size},{self.input_dim_per_device}], got {tuple(x.shape)}")
            flat_x = x.reshape(batch_size, seq_len, -1)
        elif x.dim() == 3:
            batch_size, seq_len, flat_dim = x.shape
            if flat_dim != self.input_dim:
                raise ValueError(f"Expected last dim {self.input_dim}, got {flat_dim}")
            flat_x = x
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        input_ori = extract_input_ori(x, self.combo_size, self.input_dim_per_device)
        feat = self.input_proj(flat_x)
        feat = feat + self.pos_embed[:, :seq_len]
        feat, _ = self.temporal_backbone(feat)
        feat = self.norm(feat)

        pred_acc = None
        if self.acc_head is not None:
            pred_acc = self.acc_head(feat).view(batch_size, seq_len, self.combo_size, 3)
        pred_ori6d = self.ori_head(feat).view(batch_size, seq_len, self.combo_size, 6)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)

        if seq_mask is not None:
            mask = seq_mask.view(batch_size, seq_len, 1, 1)
            if pred_acc is not None:
                pred_acc = pred_acc * mask
            pred_ori6d = pred_ori6d * mask

        if return_ori6d:
            return pred_acc, pred_ori, pred_ori6d
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame_windowed(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        if self.buffer is None:
            recent = frame.unsqueeze(0).repeat(self.online_past_frames, 1)
        else:
            recent = torch.cat([self.buffer[1:], frame.unsqueeze(0)], dim=0)

        future = frame.unsqueeze(0).repeat(self.online_future_frames, 1)
        window = torch.cat([recent, future], dim=0)
        seq = window.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=False)
        self.buffer = recent

        output_idx = self.online_past_frames - 1
        pred_acc_frame = None if pred_acc is None else pred_acc[0, output_idx]
        return pred_acc_frame, pred_ori[0, output_idx]

    @torch.no_grad()
    def forward_frame(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        feat = self.input_proj(frame.unsqueeze(0).unsqueeze(0))
        pos_idx = min(self.stream_step, self.max_seq_len - 1)
        feat = feat + self.pos_embed[:, pos_idx : pos_idx + 1]
        feat, self.hidden_state = self.temporal_backbone(feat, self.hidden_state)
        feat = self.norm(feat[:, 0])

        pred_acc_frame = None
        if self.acc_head is not None:
            pred_acc_frame = self.acc_head(feat).view(self.combo_size, 3)
        pred_ori6d = self.ori_head(feat).view(self.combo_size, 6)
        input_ori = frame.view(self.combo_size, self.input_dim_per_device)[:, 3:].reshape(self.combo_size, 3, 3)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)
        self.stream_step += 1
        return pred_acc_frame, pred_ori


class CrossDeviceTransformerResidualCalibrator(nn.Module):
    """
    Cooperative multi-device calibrator based on:
    1) per-device encoding
    2) confidence-gated cross-device fusion
    3) temporal transformer per device
    4) residual orientation calibration
    """

    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        predict_acc: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 256,
        online_past_frames: int = 45,
        online_future_frames: int = 5,
    ):
        super().__init__()
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.input_dim = combo_size * input_dim_per_device
        self.predict_acc = predict_acc
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.online_past_frames = online_past_frames
        self.online_future_frames = online_future_frames

        self.input_proj = nn.Linear(input_dim_per_device, hidden_dim)
        self.device_embed = nn.Embedding(combo_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, hidden_dim))
        self.fusion_blocks = nn.ModuleList(
            [
                ConfidenceGatedCrossDeviceFusion(
                    hidden_dim=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.acc_head = nn.Linear(hidden_dim, 3) if predict_acc else None
        self.ori_head = nn.Linear(hidden_dim, 6)
        self.buffer = None

    def reset(self):
        self.buffer = None

    def _prepare_input(self, x: torch.Tensor):
        if x.dim() == 4:
            batch_size, seq_len, combo_size, feat_dim = x.shape
            if combo_size != self.combo_size or feat_dim != self.input_dim_per_device:
                raise ValueError(
                    f"Expected [B,T,{self.combo_size},{self.input_dim_per_device}], got {tuple(x.shape)}"
                )
            return x
        if x.dim() == 3:
            batch_size, seq_len, flat_dim = x.shape
            if flat_dim != self.input_dim:
                raise ValueError(f"Expected last dim {self.input_dim}, got {flat_dim}")
            return x.view(batch_size, seq_len, self.combo_size, self.input_dim_per_device)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def _causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        return_ori6d: bool = False,
    ):
        x = self._prepare_input(x)
        batch_size, seq_len, combo_size, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        input_ori = x[..., 3:].reshape(batch_size, seq_len, combo_size, 3, 3)
        feat = self.input_proj(x)
        device_ids = torch.arange(combo_size, device=feat.device)
        feat = feat + self.device_embed(device_ids).view(1, 1, combo_size, self.hidden_dim)
        feat = feat + self.pos_embed[:, :seq_len]

        for fusion_block in self.fusion_blocks:
            feat, _ = fusion_block(feat)

        feat = feat.permute(0, 2, 1, 3).reshape(batch_size * combo_size, seq_len, self.hidden_dim)
        padding_mask = None
        if seq_mask is not None:
            padding_mask = (~seq_mask).unsqueeze(1).expand(batch_size, combo_size, seq_len).reshape(
                batch_size * combo_size, seq_len
            )
        encoder_kwargs = {"src_key_padding_mask": padding_mask}
        if causal:
            encoder_kwargs["mask"] = self._causal_mask(seq_len, feat.device)
        feat = self.temporal_encoder(feat, **encoder_kwargs)
        feat = self.norm(feat)

        pred_acc = None
        if self.acc_head is not None:
            pred_acc = self.acc_head(feat).view(batch_size, combo_size, seq_len, 3).permute(0, 2, 1, 3)
        pred_ori6d = self.ori_head(feat).view(batch_size, combo_size, seq_len, 6).permute(0, 2, 1, 3)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)
        if return_ori6d:
            return pred_acc, pred_ori, pred_ori6d
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        self.buffer = frame.unsqueeze(0) if self.buffer is None else torch.cat([self.buffer, frame.unsqueeze(0)], dim=0)
        if self.buffer.shape[0] > self.max_seq_len:
            self.buffer = self.buffer[-self.max_seq_len:]

        seq = self.buffer.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=True)
        pred_acc_frame = None if pred_acc is None else pred_acc[0, -1]
        return pred_acc_frame, pred_ori[0, -1]

    @torch.no_grad()
    def forward_frame_windowed(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        if self.buffer is None:
            recent = frame.unsqueeze(0).repeat(self.online_past_frames, 1)
        else:
            recent = torch.cat([self.buffer[1:], frame.unsqueeze(0)], dim=0)

        future = frame.unsqueeze(0).repeat(self.online_future_frames, 1)
        window = torch.cat([recent, future], dim=0)
        seq = window.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=False)
        self.buffer = recent

        output_idx = self.online_past_frames - 1
        pred_acc_frame = None if pred_acc is None else pred_acc[0, output_idx]
        return pred_acc_frame, pred_ori[0, output_idx]


class CrossDeviceTransformerCalibrator(nn.Module):
    """
    Cooperative multi-device calibrator with direct absolute orientation output.

    This keeps the same cross-device attention + confidence-gated fusion design
    as `CrossDeviceTransformerResidualCalibrator`, but predicts calibrated
    orientations directly instead of residual corrections.
    """

    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        predict_acc: bool = False,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 256,
        online_past_frames: int = 45,
        online_future_frames: int = 5,
    ):
        super().__init__()
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.input_dim = combo_size * input_dim_per_device
        self.predict_acc = predict_acc
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.online_past_frames = online_past_frames
        self.online_future_frames = online_future_frames

        self.input_proj = nn.Linear(input_dim_per_device, hidden_dim)
        self.device_embed = nn.Embedding(combo_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, hidden_dim))
        self.fusion_blocks = nn.ModuleList(
            [
                ConfidenceGatedCrossDeviceFusion(
                    hidden_dim=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.acc_head = nn.Linear(hidden_dim, 3) if predict_acc else None
        self.ori_head = nn.Linear(hidden_dim, 6)
        self.buffer = None

    def reset(self):
        self.buffer = None

    def _prepare_input(self, x: torch.Tensor):
        if x.dim() == 4:
            batch_size, seq_len, combo_size, feat_dim = x.shape
            if combo_size != self.combo_size or feat_dim != self.input_dim_per_device:
                raise ValueError(
                    f"Expected [B,T,{self.combo_size},{self.input_dim_per_device}], got {tuple(x.shape)}"
                )
            return x
        if x.dim() == 3:
            batch_size, seq_len, flat_dim = x.shape
            if flat_dim != self.input_dim:
                raise ValueError(f"Expected last dim {self.input_dim}, got {flat_dim}")
            return x.view(batch_size, seq_len, self.combo_size, self.input_dim_per_device)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def _causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        return_ori6d: bool = False,
    ):
        x = self._prepare_input(x)
        batch_size, seq_len, combo_size, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        feat = self.input_proj(x)
        device_ids = torch.arange(combo_size, device=feat.device)
        feat = feat + self.device_embed(device_ids).view(1, 1, combo_size, self.hidden_dim)
        feat = feat + self.pos_embed[:, :seq_len]

        for fusion_block in self.fusion_blocks:
            feat, _ = fusion_block(feat)

        feat = feat.permute(0, 2, 1, 3).reshape(batch_size * combo_size, seq_len, self.hidden_dim)
        padding_mask = None
        if seq_mask is not None:
            padding_mask = (~seq_mask).unsqueeze(1).expand(batch_size, combo_size, seq_len).reshape(
                batch_size * combo_size, seq_len
            )
        encoder_kwargs = {"src_key_padding_mask": padding_mask}
        if causal:
            encoder_kwargs["mask"] = self._causal_mask(seq_len, feat.device)
        feat = self.temporal_encoder(feat, **encoder_kwargs)
        feat = self.norm(feat)

        pred_acc = None
        if self.acc_head is not None:
            pred_acc = self.acc_head(feat).view(batch_size, combo_size, seq_len, 3).permute(0, 2, 1, 3)
        pred_ori6d = self.ori_head(feat).view(batch_size, combo_size, seq_len, 6).permute(0, 2, 1, 3)
        pred_ori = art.math.r6d_to_rotation_matrix(pred_ori6d.reshape(-1, 6)).view(
            batch_size, seq_len, combo_size, 3, 3
        )
        if return_ori6d:
            return pred_acc, pred_ori, pred_ori6d
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        self.buffer = frame.unsqueeze(0) if self.buffer is None else torch.cat([self.buffer, frame.unsqueeze(0)], dim=0)
        if self.buffer.shape[0] > self.max_seq_len:
            self.buffer = self.buffer[-self.max_seq_len:]

        seq = self.buffer.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=True)
        pred_acc_frame = None if pred_acc is None else pred_acc[0, -1]
        return pred_acc_frame, pred_ori[0, -1]

    @torch.no_grad()
    def forward_frame_windowed(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        if self.buffer is None:
            recent = frame.unsqueeze(0).repeat(self.online_past_frames, 1)
        else:
            recent = torch.cat([self.buffer[1:], frame.unsqueeze(0)], dim=0)

        future = frame.unsqueeze(0).repeat(self.online_future_frames, 1)
        window = torch.cat([recent, future], dim=0)
        seq = window.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=False)
        self.buffer = recent

        output_idx = self.online_past_frames - 1
        pred_acc_frame = None if pred_acc is None else pred_acc[0, output_idx]
        return pred_acc_frame, pred_ori[0, output_idx]


class DeviceCalibrator(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.acc_head = nn.Linear(hidden_dim, 3)
        self.ori_head = nn.Linear(hidden_dim, 6)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.acc_head(feat), self.ori_head(feat)


class MultiDeviceIMUCalibrator(nn.Module):
    def __init__(self, num_devices: int = 7, input_dim: int = 12, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_devices = num_devices
        self.input_dim = input_dim
        self.device_calibrators = nn.ModuleList(
            [DeviceCalibrator(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_devices)]
        )

    def forward(self, x: torch.Tensor):
        # x: [B, D, C]
        pred_acc, pred_ori6d = [], []
        for device_idx, calibrator in enumerate(self.device_calibrators):
            acc_i, ori_i = calibrator(x[:, device_idx])
            pred_acc.append(acc_i)
            pred_ori6d.append(ori_i)
        pred_acc = torch.stack(pred_acc, dim=1)
        pred_ori6d = torch.stack(pred_ori6d, dim=1)
        pred_ori = art.math.r6d_to_rotation_matrix(pred_ori6d.reshape(-1, 6)).view(-1, self.num_devices, 3, 3)
        return pred_acc, pred_ori


class TemporalIMUCalibrator(nn.Module):
    def __init__(
        self,
        num_devices: int = 7,
        input_dim: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.num_devices = num_devices
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.device_embed = nn.Embedding(num_devices, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.acc_head = nn.Linear(hidden_dim, 3)
        self.ori_head = nn.Linear(hidden_dim, 6)
        self.buffer = None

    def reset(self):
        self.buffer = None

    def _causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x: torch.Tensor, seq_mask: Optional[torch.Tensor] = None):
        # x: [B, T, D, C], seq_mask: [B, T]
        batch_size, seq_len, num_devices, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        device_ids = torch.arange(num_devices, device=x.device)
        feat = self.input_proj(x)
        feat = feat + self.device_embed(device_ids).view(1, 1, num_devices, self.hidden_dim)
        feat = feat + self.pos_embed[:, :seq_len].unsqueeze(2)

        feat = feat.permute(0, 2, 1, 3).reshape(batch_size * num_devices, seq_len, self.hidden_dim)
        padding_mask = None
        if seq_mask is not None:
            padding_mask = (~seq_mask).unsqueeze(1).expand(batch_size, num_devices, seq_len)
            padding_mask = padding_mask.reshape(batch_size * num_devices, seq_len)

        feat = self.encoder(
            feat,
            mask=self._causal_mask(seq_len, feat.device),
            src_key_padding_mask=padding_mask,
        )
        feat = self.norm(feat)

        pred_acc = self.acc_head(feat)
        pred_ori6d = self.ori_head(feat)
        pred_ori = art.math.r6d_to_rotation_matrix(pred_ori6d.reshape(-1, 6)).view(
            batch_size, num_devices, seq_len, 3, 3
        )

        pred_acc = pred_acc.view(batch_size, num_devices, seq_len, 3).permute(0, 2, 1, 3)
        pred_ori = pred_ori.permute(0, 2, 1, 3, 4)
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame(self, frame: torch.Tensor):
        # frame: [D, C]
        if frame.dim() != 2:
            raise ValueError(f"Expected frame shape [D, C], got {tuple(frame.shape)}")
        self.buffer = frame.unsqueeze(0) if self.buffer is None else torch.cat([self.buffer, frame.unsqueeze(0)], dim=0)
        if self.buffer.shape[0] > self.max_seq_len:
            self.buffer = self.buffer[-self.max_seq_len:]

        seq = self.buffer.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask)
        return pred_acc[0, -1], pred_ori[0, -1]

class ComboTemporalIMUCalibrator(nn.Module):
    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        predict_acc: bool = True,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 256,
        online_past_frames: int = 45,
        online_future_frames: int = 5,
    ):
        super().__init__()
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.input_dim = combo_size * input_dim_per_device
        self.predict_acc = predict_acc
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.online_past_frames = online_past_frames
        self.online_future_frames = online_future_frames

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.acc_head = nn.Linear(hidden_dim, combo_size * 3) if predict_acc else None
        self.ori_head = nn.Linear(hidden_dim, combo_size * 6)
        self.buffer = None

    def reset(self):
        self.buffer = None

    def _causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
        return_ori6d: bool = False,
    ):
        # x: [B, T, combo_size, 12] or [B, T, combo_size*12]
        if x.dim() == 4:
            batch_size, seq_len, combo_size, feat_dim = x.shape
            if combo_size != self.combo_size or feat_dim != self.input_dim_per_device:
                raise ValueError(f"Expected [B,T,{self.combo_size},{self.input_dim_per_device}], got {tuple(x.shape)}")
            flat_x = x.reshape(batch_size, seq_len, -1)
        elif x.dim() == 3:
            batch_size, seq_len, flat_dim = x.shape
            if flat_dim != self.input_dim:
                raise ValueError(f"Expected last dim {self.input_dim}, got {flat_dim}")
            flat_x = x
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        input_ori = extract_input_ori(x, self.combo_size, self.input_dim_per_device)
        feat = self.input_proj(flat_x)
        feat = feat + self.pos_embed[:, :seq_len]

        padding_mask = None
        if seq_mask is not None:
            padding_mask = ~seq_mask

        encoder_kwargs = {"src_key_padding_mask": padding_mask}
        if causal:
            encoder_kwargs["mask"] = self._causal_mask(seq_len, feat.device)
        feat = self.encoder(feat, **encoder_kwargs)
        feat = self.norm(feat)

        pred_acc = None
        if self.acc_head is not None:
            pred_acc = self.acc_head(feat).view(batch_size, seq_len, self.combo_size, 3)
        pred_ori6d = self.ori_head(feat).view(batch_size, seq_len, self.combo_size, 6)
        pred_ori, _ = compose_residual_orientation(input_ori, pred_ori6d)
        if return_ori6d:
            return pred_acc, pred_ori, pred_ori6d
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame(self, frame: torch.Tensor):
        # frame: [combo_size, 12] or [combo_size*12]
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        self.buffer = frame.unsqueeze(0) if self.buffer is None else torch.cat([self.buffer, frame.unsqueeze(0)], dim=0)
        if self.buffer.shape[0] > self.max_seq_len:
            self.buffer = self.buffer[-self.max_seq_len:]

        seq = self.buffer.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask)
        pred_acc_frame = None if pred_acc is None else pred_acc[0, -1]
        return pred_acc_frame, pred_ori[0, -1]

    @torch.no_grad()
    def forward_frame_windowed(self, frame: torch.Tensor):
        # frame: [combo_size, 12] or [combo_size*12]
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        if self.buffer is None:
            recent = frame.unsqueeze(0).repeat(self.online_past_frames, 1)
        else:
            recent = torch.cat([self.buffer[1:], frame.unsqueeze(0)], dim=0)

        future = frame.unsqueeze(0).repeat(self.online_future_frames, 1)
        window = torch.cat([recent, future], dim=0)
        seq = window.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=False)
        self.buffer = recent

        output_idx = self.online_past_frames - 1
        pred_acc_frame = None if pred_acc is None else pred_acc[0, output_idx]
        return pred_acc_frame, pred_ori[0, output_idx]


class TICComboCalibrator(nn.Module):
    """
    TIC-style transformer baseline adapted to the current combo-calibration task.

    Unlike the original TIC repo, this model keeps our existing framewise
    real->synthetic supervision and online windowed inference, but swaps in a
    lighter transformer backbone with a TIC-like pooled context branch.
    """

    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        predict_acc: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 4,
        nhead: int = 8,
        max_seq_len: int = 256,
        online_past_frames: int = 45,
        online_future_frames: int = 5,
    ):
        super().__init__()
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.input_dim = combo_size * input_dim_per_device
        self.predict_acc = predict_acc
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.online_past_frames = online_past_frames
        self.online_future_frames = online_future_frames

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.acc_head = nn.Linear(hidden_dim, combo_size * 3) if predict_acc else None
        self.ori_head = nn.Linear(hidden_dim, combo_size * 6)
        self.buffer = None

    def reset(self):
        self.buffer = None

    def _causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        return_ori6d: bool = False,
    ):
        if x.dim() == 4:
            batch_size, seq_len, combo_size, feat_dim = x.shape
            if combo_size != self.combo_size or feat_dim != self.input_dim_per_device:
                raise ValueError(f"Expected [B,T,{self.combo_size},{self.input_dim_per_device}], got {tuple(x.shape)}")
            flat_x = x.reshape(batch_size, seq_len, -1)
        elif x.dim() == 3:
            batch_size, seq_len, flat_dim = x.shape
            if flat_dim != self.input_dim:
                raise ValueError(f"Expected last dim {self.input_dim}, got {flat_dim}")
            flat_x = x
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        feat = self.input_proj(flat_x)
        feat = feat + self.pos_embed[:, :seq_len]

        padding_mask = None if seq_mask is None else ~seq_mask
        encoder_kwargs = {"src_key_padding_mask": padding_mask}
        if causal:
            encoder_kwargs["mask"] = self._causal_mask(seq_len, feat.device)
        feat = self.encoder(feat, **encoder_kwargs)
        feat = self.norm(feat)

        if seq_mask is None:
            pooled = feat.mean(dim=1)
        else:
            denom = seq_mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (feat * seq_mask.unsqueeze(-1)).sum(dim=1) / denom
        context = self.context_mlp(pooled).unsqueeze(1)
        fused = feat + context

        pred_acc = None
        if self.acc_head is not None:
            pred_acc = self.acc_head(fused).view(batch_size, seq_len, self.combo_size, 3)
        pred_ori6d = self.ori_head(fused).view(batch_size, seq_len, self.combo_size, 6)
        pred_ori = art.math.r6d_to_rotation_matrix(pred_ori6d.reshape(-1, 6)).view(
            batch_size, seq_len, self.combo_size, 3, 3
        )
        if return_ori6d:
            return pred_acc, pred_ori, pred_ori6d
        return pred_acc, pred_ori

    @torch.no_grad()
    def forward_frame_windowed(self, frame: torch.Tensor):
        if frame.dim() == 2:
            frame = frame.reshape(-1)
        elif frame.dim() != 1:
            raise ValueError(f"Expected frame shape [combo_size, 12] or [combo_size*12], got {tuple(frame.shape)}")
        if frame.shape[0] != self.input_dim:
            raise ValueError(f"Expected flattened frame dim {self.input_dim}, got {frame.shape[0]}")

        if self.buffer is None:
            recent = frame.unsqueeze(0).repeat(self.online_past_frames, 1)
        else:
            recent = torch.cat([self.buffer[1:], frame.unsqueeze(0)], dim=0)

        future = frame.unsqueeze(0).repeat(self.online_future_frames, 1)
        window = torch.cat([recent, future], dim=0)
        seq = window.unsqueeze(0)
        seq_mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=seq.device)
        pred_acc, pred_ori = self(seq, seq_mask=seq_mask, causal=False)
        self.buffer = recent

        output_idx = self.online_past_frames - 1
        pred_acc_frame = None if pred_acc is None else pred_acc[0, output_idx]
        return pred_acc_frame, pred_ori[0, output_idx]
