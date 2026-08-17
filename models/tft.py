import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodingTFT(nn.Module):
    """Positional encoding for temporal fusion."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network: produces interpretable feature attention weights
    per feature per timestep using gated linear units.

    Reference: Lim et al. (2021), Temporal Fusion Transformer
    """

    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.gating = nn.Linear(n_features, hidden_size)
        self.selection = nn.Linear(n_features, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, n_features)
        Returns:
            weighted_features: (batch, seq_len, hidden_size)
            selection_weights: (batch, seq_len, n_features) — interpretable
        """
        gate = torch.sigmoid(self.gating(x))
        raw = torch.tanh(self.selection(x))
        weighted = gate * raw

        n_feat = x.shape[-1]
        feat_weights = (gate.abs().mean(dim=-1, keepdim=True) / n_feat)
        selection_weights = torch.softmax(feat_weights * n_feat, dim=-1)

        return self.dropout(weighted), selection_weights


class StaticEncoder(nn.Module):
    """Encodes static (known-in-advance) features via gating.

    Produces a static context vector that is added to the temporal decoder output.
    """

    def __init__(self, n_static: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.gating = nn.Linear(n_static, hidden_size)
        self.selection = nn.Linear(n_static, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_static: (batch, n_static)
        Returns:
            static_context: (batch, hidden_size)
            static_weights: (batch, n_static)
        """
        gate = torch.sigmoid(self.gating(x_static))
        raw = torch.tanh(self.selection(x_static))
        context = gate * raw

        n_feat = x_static.shape[-1]
        feat_weights = (gate.abs().mean(dim=-1, keepdim=True) / n_feat)
        static_weights = torch.softmax(feat_weights * n_feat, dim=-1)

        return self.dropout(context), static_weights


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention for the decoder.

    Produces interpretable attention weights over the encoder sequence.
    """

    def __init__(self, query_dim: int, key_dim: int, head_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, head_dim)
        self.key_proj = nn.Linear(key_dim, head_dim)
        self.v_proj = nn.Linear(key_dim, head_dim)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: (batch, query_dim)
        keys: (batch, seq_len, key_dim)
        Returns:
            context: (batch, head_dim)
            attn_weights: (batch, seq_len)
        """
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(keys)
        v = self.v_proj(keys)

        attn_scores = torch.bmm(q, k.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        seq_attn = torch.bmm(attn_weights, v).squeeze(1)
        return seq_attn, attn_weights.squeeze(1)


class TFTModel(nn.Module):
    """Temporal Fusion Transformer for multi-task financial time-series forecasting.

    Architecture:
        Input → Variable Selection → Static Encoder → GRU Decoder
            → Additive Attention → Multi-Task Heads

    Multi-task heads:
        - Regression: predicted return
        - Binary classification: P(UP)
        - Ternary classification: P(DOWN/SIDEWAYS/UP)
        - Uncertainty: MC Dropout variance

    Reference: Lim et al. (2021), "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting"
    """

    def __init__(
        self,
        input_size: int,
        n_static_features: int,
        hidden_size: int = 64,
        num_gru_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        seq_length: int = 60,
        task: str = "regression",
        n_classes: int = 2,
        mc_dropout_samples: int = 10,
    ):
        super().__init__()
        self.task = task
        self.seq_length = seq_length
        self.input_size = input_size
        self.n_classes = n_classes
        self.n_static = n_static_features
        self.mc_dropout_samples = mc_dropout_samples
        self.hidden_size = hidden_size

        n_timevarying = input_size - n_static_features
        if n_timevarying < 1:
            raise ValueError(
                f"n_static_features ({n_static_features}) >= input_size ({input_size}). "
                "Need at least 1 time-varying feature."
            )

        self.static_encoder = StaticEncoder(n_static_features, hidden_size, dropout)

        self.var_selection = VariableSelectionNetwork(n_timevarying, hidden_size, dropout)

        gru_input_dim = hidden_size * 2
        self.gru = nn.GRU(
            gru_input_dim, hidden_size, num_gru_layers,
            batch_first=True, dropout=dropout if num_gru_layers > 1 else 0,
        )

        self.pos_encoder = PositionalEncodingTFT(gru_input_dim, dropout=dropout)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=gru_input_dim, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )

        self.additive_attention = AdditiveAttention(hidden_size, hidden_size, hidden_size)

        self.fc_dropout = nn.Dropout(dropout)

        output_dim = hidden_size * 3

        self.reg_head = nn.Sequential(
            nn.Linear(output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.binary_head = nn.Sequential(
            nn.Linear(output_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.ternary_head = nn.Sequential(
            nn.Linear(output_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def _split_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input into time-varying and static features.

        Convention: last n_static features are static.
        x: (batch, seq_len, input_size)
        Returns:
            x_timevarying: (batch, seq_len, input_size - n_static)
            x_static: (batch, n_static) — taken from last timestep
        """
        x_static = x[:, -1, -self.n_static:]
        x_timevarying = x[:, :, :-self.n_static]
        return x_timevarying, x_static

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, seq_len, input_size)
        Returns dict with:
            'regression': (batch, 1)
            'binary': (batch, 1)
            'ternary': (batch, n_classes)
        """
        x_timevarying, x_static = self._split_inputs(x)

        static_context, static_weights = self.static_encoder(x_static)

        var_selected, var_weights = self.var_selection(x_timevarying)

        decoder_input = torch.cat([var_selected, static_context.unsqueeze(1).expand(-1, x_timevarying.size(1), -1)], dim=-1)

        pos_encoded = self.pos_encoder(decoder_input)

        attn_out, _ = self.self_attn(pos_encoded, pos_encoded, pos_encoded)
        attn_out = self.fc_dropout(attn_out)

        gru_out, _ = self.gru(attn_out)

        last_hidden = gru_out[:, -1, :]
        encoder_states = gru_out

        context_vec, temporal_weights = self.additive_attention(last_hidden, encoder_states)

        combined = torch.cat([last_hidden, context_vec, static_context], dim=-1)

        reg_out = self.reg_head(combined)
        binary_out = self.binary_head(combined)
        ternary_out = self.ternary_head(combined)

        return {
            "regression": reg_out,
            "binary": binary_out,
            "ternary": ternary_out,
        }

    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run inference with Monte Carlo Dropout for uncertainty estimation.

        Returns mean predictions and variances for regression output.
        """
        self.train()

        reg_preds = []
        binary_preds = []
        ternary_preds = []

        for _ in range(self.mc_dropout_samples):
            with torch.no_grad():
                out = self.forward(x)
                reg_preds.append(out["regression"].cpu())
                binary_preds.append(out["binary"].cpu())
                ternary_preds.append(out["ternary"].cpu())

        self.eval()

        reg_stack = torch.stack(reg_preds)
        binary_stack = torch.stack(binary_preds)
        ternary_stack = torch.stack(ternary_preds)

        return {
            "regression_mean": reg_stack.mean(dim=0),
            "regression_std": reg_stack.std(dim=0),
            "binary_mean": binary_stack.mean(dim=0),
            "binary_std": binary_stack.std(dim=0),
            "ternary_mean": ternary_stack.mean(dim=0),
            "ternary_std": ternary_stack.std(dim=0),
        }


# ─── Loss Functions ──────────────────────────────────────────────────────────


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Learns per-task log-variance parameters (ρ) to automatically balance
    loss contributions during training. Tasks with higher inherent variance
    get lower weight.

    L = sum_i [ L_i / (2 * exp(ρ_i)) + ρ_i ]
    """

    def __init__(self, task: str, n_classes: int = 2):
        super().__init__()
        self.task = task
        self.n_classes = n_classes

        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_binary = nn.Parameter(torch.zeros(1))
        self.log_var_ternary = nn.Parameter(torch.zeros(1))

        self.criterion_reg = nn.MSELoss(reduction="mean")
        self.criterion_binary = nn.BCEWithLogitsLoss(reduction="mean")
        self.criterion_ternary = nn.CrossEntropyLoss(reduction="mean")

    def _get_device(self, x: torch.Tensor) -> torch.device:
        """Get device from any tensor in the module."""
        return next(self.parameters()).device

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        y_reg: torch.Tensor,
        y_binary: torch.Tensor,
        y_ternary: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor
            losses: dict with individual loss values (for logging)
        """
        prec_reg = torch.exp(-self.log_var_reg)
        prec_binary = torch.exp(-self.log_var_binary)
        prec_ternary = torch.exp(-self.log_var_ternary)

        l_reg = self.criterion_reg(outputs["regression"].squeeze(-1), y_reg.float())
        loss_reg = l_reg * prec_reg + self.log_var_reg

        if self.n_classes == 2:
            l_binary = self.criterion_binary(
                outputs["binary"].squeeze(-1), y_binary.float()
            )
            if class_weights is not None:
                raw_binary = F.binary_cross_entropy_with_logits(
                    outputs["binary"].squeeze(-1), y_binary.float(), reduction="none"
                )
                l_binary = (raw_binary * class_weights).mean()
            loss_binary = l_binary * prec_binary + self.log_var_binary
        else:
            loss_binary = torch.tensor(0.0, device=y_reg.device)

        if y_ternary is not None and self.n_classes > 2:
            l_ternary = self.criterion_ternary(outputs["ternary"], y_ternary.long())
            loss_ternary = l_ternary * prec_ternary + self.log_var_ternary
        else:
            loss_ternary = torch.tensor(0.0, device=y_reg.device)

        total = loss_reg + loss_binary + loss_ternary

        losses = {
            "loss_reg": float(l_reg.item()),
            "loss_binary": float(l_binary.item()) if self.n_classes == 2 else 0.0,
            "loss_ternary": float(l_ternary.item()) if y_ternary is not None and self.n_classes > 2 else 0.0,
            "prec_reg": float(prec_reg.item()),
            "prec_binary": float(prec_binary.item()),
            "prec_ternary": float(prec_ternary.item()),
        }
        return total, losses
