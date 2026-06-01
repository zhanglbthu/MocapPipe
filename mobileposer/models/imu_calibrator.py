import torch
import torch.nn as nn
from typing import Optional

import articulate as art


def build_imu_input(acc: torch.Tensor, ori: torch.Tensor):
    ori9d = ori.reshape(ori.shape[0], ori.shape[1], 9)
    return torch.cat([acc, ori9d], dim=-1)


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
        pred_ori = art.math.r6d_to_rotation_matrix(pred_ori6d.reshape(-1, 6)).view(
            batch_size, seq_len, self.combo_size, 3, 3
        )
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
