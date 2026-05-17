"""
Model architectures for stock trend prediction.
- MLP baseline
- GRU + Multi-Head Attention (primary model)
- Transformer encoder (comparison)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT, SEQ_LEN


class PositionalEncoding(nn.Module):
    """Learned positional encoding for time series."""
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]


class MLPBaseline(nn.Module):
    """Simple MLP baseline: flatten the entire window."""
    def __init__(self, input_dim, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.flatten_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: [B, seq_len, F]
        x = x.reshape(x.size(0), -1)
        return self.net(x)


class GRUAttention(nn.Module):
    """
    GRU with Multi-Head Self-Attention.
    Architecture:
        Input Projection → Positional Encoding → Bi-GRU → Self-Attention →
        Concat Pooling → MLP Head → Output
    """
    def __init__(
        self,
        input_dim,
        d_model=D_MODEL,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(seq_len, d_model)

        # Bi-GRU layers
        self.gru1 = nn.GRU(
            d_model, hidden_dim // 2, num_layers=1,
            bidirectional=True, batch_first=True, dropout=0.0,
        )  # output: [B, L, hidden_dim]

        self.gru2 = nn.GRU(
            hidden_dim, hidden_dim // 4, num_layers=1,
            bidirectional=True, batch_first=True, dropout=0.0,
        )  # output: [B, L, hidden_dim//2]

        self.gru_dropout = nn.Dropout(dropout)

        # Multi-head self-attention
        gru_out_dim = hidden_dim // 2
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(gru_out_dim)

        # MLP head (after pooling)
        head_input = gru_out_dim * 2  # concat of mean pooling + max pooling
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [B, L, F]
        B, L, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # [B, L, d_model]

        # Positional encoding
        x = self.pos_enc(x)

        # GRU layers
        x, _ = self.gru1(x)  # [B, L, hidden_dim]
        x = self.gru_dropout(x)
        x, _ = self.gru2(x)  # [B, L, hidden_dim//2]
        x = self.gru_dropout(x)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # [B, L, hidden_dim//2]
        x = self.attn_norm(x + attn_out)  # residual + norm

        # Pooling
        avg_pool = x.mean(dim=1)  # [B, hidden_dim//2]
        max_pool = x.max(dim=1).values  # [B, hidden_dim//2]
        pooled = torch.cat([avg_pool, max_pool], dim=-1)  # [B, hidden_dim]

        # MLP head
        out = self.head(pooled)  # [B, 1]
        return out


class TransformerEncoder(nn.Module):
    """
    Pure Transformer encoder for time series.
    """
    def __init__(
        self,
        input_dim,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
        num_tf_layers=4,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_tf_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)  # [B, L, d_model]

        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        pooled = torch.cat([avg_pool, max_pool], dim=-1)
        out = self.head(pooled)
        return out


def create_model(model_name: str, input_dim: int) -> nn.Module:
    """Factory function for model creation."""
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLPBaseline(input_dim)
    elif model_name == "gru_attention":
        return GRUAttention(input_dim)
    elif model_name == "transformer":
        return TransformerEncoder(input_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
