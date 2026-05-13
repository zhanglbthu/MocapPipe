import torch
import torch.nn as nn

from config import datasets


class TransformerBackbone(nn.Module):
    """
    Transformer encoder with learned positional embeddings.

    Default dimensions are chosen to match the current bidirectional LSTM
    backbone parameter count exactly for DirectPoserNet.
    """

    def __init__(
        self,
        n_input,
        n_output,
        d_model=184,
        nhead=8,
        num_layers=4,
        dim_feedforward=1420,
        dropout=0.4,
        max_len=None,
    ):
        super().__init__()
        self.max_len = max_len or datasets.window_length
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, d_model))
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_output)

    def forward(self, x, seq_lengths=None):
        if x.shape[1] > self.max_len:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_len={self.max_len}")

        data = self.input_proj(x)
        data = data + self.pos_embed[:, : x.shape[1]]
        data = self.dropout(data)

        padding_mask = None
        if seq_lengths is not None:
            steps = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            lengths = torch.as_tensor(seq_lengths, device=x.device).unsqueeze(1)
            padding_mask = steps >= lengths

        data = self.encoder(data, src_key_padding_mask=padding_mask)
        data = self.output_proj(data)
        return data, seq_lengths, None
