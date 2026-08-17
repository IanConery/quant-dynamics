import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

from config.settings import DEEP_PARAMS, DEVICE
from utils.logger import setup_logger

logger = setup_logger(__name__)

from models.tft import TFTModel, MultiTaskLoss


class PositionalEncoding(nn.Module):
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

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 dropout: float = 0.3, task: str = "regression", n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        output_size = n_classes if (task == "classification" and n_classes > 2) else 1
        self.fc = nn.Linear(hidden_size, output_size)
        self.task = task
        self.n_classes = n_classes

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        if self.task == "classification" and self.n_classes == 2:
            out = torch.sigmoid(out)
        elif self.task == "classification" and self.n_classes > 2:
            out = torch.softmax(out, dim=-1)
        return out


class iTransformerModel(nn.Module):
    """
    Inverted Transformer: features become tokens, timesteps become the sequence dimension.
    Reference: iTransformer (Liu et al., 2023)
    """

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.2,
                 seq_length: int = 60, task: str = "regression", n_classes: int = 2):
        super().__init__()
        self.task = task
        self.seq_length = seq_length
        self.input_size = input_size
        self.n_classes = n_classes

        self.enc = nn.Linear(seq_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_encoder = PositionalEncoding(d_model, max_len=input_size + 10, dropout=dropout)
        self.dec = nn.Linear(d_model, seq_length)
        output_size = n_classes if (task == "classification" and n_classes > 2) else 1
        self.head = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        """
        batch_size = x.size(0)

        x_transposed = x.transpose(1, 2)

        embed = self.enc(x_transposed)

        pos_embed = self.pos_encoder(embed)

        attn_out = self.transformer_encoder(pos_embed)

        dec_out = self.dec(attn_out)

        x_decoded = dec_out.transpose(1, 2)

        pooled = x_decoded.mean(dim=1)

        out = self.head(pooled)

        if self.task == "classification" and self.n_classes == 2:
            out = torch.sigmoid(out)
        elif self.task == "classification" and self.n_classes > 2:
            out = torch.softmax(out, dim=-1)
        return out


# ─── DeepModel Wrapper ─────────────────────────────────────────────────────


class DeepModel:
    def __init__(self, model_type: str, input_size: int, task: str,
                 params: Dict, device: str = None, n_classes: int = 2):
        self.model_type = model_type
        self.task = task
        self.params = params
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._instantiate(input_size)
        self.model.to(self.device)

    def _instantiate(self, input_size: int) -> nn.Module:
        if self.model_type == "lstm":
            p = self.params
            return LSTMModel(
                input_size=input_size,
                hidden_size=p.get("hidden_size", 128),
                num_layers=p.get("num_layers", 3),
                dropout=p.get("dropout", 0.3),
                task=self.task,
                n_classes=self.n_classes,
            )
        elif self.model_type == "itransformer":
            p = self.params
            return iTransformerModel(
                input_size=input_size,
                d_model=p.get("d_model", 128),
                nhead=p.get("nhead", 8),
                num_layers=p.get("num_layers", 4),
                dim_feedforward=p.get("dim_feedforward", 512),
                dropout=p.get("dropout", 0.2),
                seq_length=p.get("seq_length", 60),
                task=self.task,
                n_classes=self.n_classes,
            )
        elif self.model_type == "tft":
            p = self.params
            n_static = p.get("n_static_features", 8)
            return TFTModel(
                input_size=input_size,
                n_static_features=n_static,
                hidden_size=p.get("hidden_size", 64),
                num_gru_layers=p.get("num_gru_layers", 2),
                nhead=p.get("nhead", 4),
                dropout=p.get("dropout", 0.1),
                seq_length=p.get("seq_length", 60),
                task=self.task,
                n_classes=self.n_classes,
                mc_dropout_samples=p.get("mc_dropout_samples", 10),
            )
        raise ValueError(f"Unknown deep model type: {self.model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            y_ternary_train: np.ndarray = None,
            y_ternary_val: np.ndarray = None) -> "DeepModel":
        p = self.params
        lr = p.get("learning_rate", 0.001)
        batch_size = p.get("batch_size", 64)
        epochs = p.get("epochs", 200)
        patience = p.get("patience", 20)
        weight_decay = p.get("weight_decay", 1e-5)

        is_tft = self.model_type == "tft"

        if is_tft:
            return self._fit_tft(
                X_train, y_train, X_val, y_val,
                y_ternary_train, y_ternary_val,
                lr, batch_size, epochs, patience, weight_decay,
            )

        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.FloatTensor(y_train).to(self.device)
        X_vl = torch.FloatTensor(X_val).to(self.device)
        y_vl = torch.FloatTensor(y_val).to(self.device)

        train_ds = TensorDataset(X_tr, y_tr)
        val_ds = TensorDataset(X_vl, y_vl)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        if self.task == "regression":
            criterion = nn.MSELoss()
            class_weights = None
        elif self.n_classes == 2:
            n_pos = float(np.sum(y_train == 1))
            n_neg = float(len(y_train) - n_pos)
            pos_weight = max(n_neg / max(n_pos, 1), 1.0)
            sample_weights = torch.where(
                y_tr == 1,
                torch.tensor(pos_weight, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            criterion = nn.BCELoss(reduction="none")
            class_weights = sample_weights
        else:
            # Multi-class: CrossEntropyLoss
            y_tr_long = y_tr.long()
            criterion = nn.CrossEntropyLoss()
            class_weights = None
            # Replace y_tr for training
            y_tr = y_tr_long
            y_vl = y_vl.long()
            train_ds = TensorDataset(X_tr, y_tr)
            val_ds = TensorDataset(X_vl, y_vl)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_loss = float("inf")
        best_state = None
        wait = 0

        logger.info(f"  {self.model_type}/{self.task}: training (epochs={epochs}, patience={patience})")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                out = self.model(X_batch).squeeze(-1)
                if self.n_classes > 2:
                    # CrossEntropyLoss expects raw logits
                    raw_loss = criterion(out, y_batch)
                    loss = raw_loss
                else:
                    raw_loss = criterion(out, y_batch.float())
                    if class_weights is not None:
                        batch_w = class_weights[:y_batch.size(0)]
                        loss = (raw_loss * batch_w).mean()
                    else:
                        loss = raw_loss.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_ds)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    out = self.model(X_batch).squeeze(-1)
                    if self.n_classes > 2:
                        val_loss += criterion(out, y_batch.long()).item() * X_batch.size(0)
                    else:
                        val_loss += criterion(out, y_batch.float()).mean().item() * X_batch.size(0)
            val_loss /= len(val_ds) if len(val_ds) > 0 else 1

            scheduler.step(val_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"    Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"    Early stopping at epoch {epoch+1}, best val_loss: {best_loss:.6f}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def _fit_tft(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        y_ternary_train: np.ndarray, y_ternary_val: np.ndarray,
        lr: float, batch_size: int, epochs: int, patience: int, weight_decay: float,
    ) -> "DeepModel":
        """TFT-specific training loop.

        If ternary data is provided: full multi-task training (Kendall 2018 loss).
        If ternary data is absent: regression-only training (MSE) for Optuna tuning.
        """
        has_ternary = y_ternary_train is not None and len(y_ternary_train) > 0

        if not has_ternary:
            return self._fit_tft_regression_only(
                X_train, y_train, X_val, y_val,
                lr, batch_size, epochs, patience, weight_decay,
            )

        y_ter_tr = torch.LongTensor(y_ternary_train).to(self.device)
        y_ter_vl = torch.LongTensor(y_ternary_val).to(self.device)

        return self._fit_tft_multitask(
            X_train, y_train, X_val, y_val,
            y_ter_tr, y_ter_vl,
            lr, batch_size, epochs, patience, weight_decay,
        )

    def _fit_tft_regression_only(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        lr: float, batch_size: int, epochs: int, patience: int, weight_decay: float,
    ) -> "DeepModel":
        """Regression-only TFT training for Optuna tuning."""
        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_tr = torch.FloatTensor(y_train).to(self.device)
        X_vl = torch.FloatTensor(X_val).to(self.device)
        y_vl = torch.FloatTensor(y_val).to(self.device)

        train_ds = TensorDataset(X_tr, y_tr)
        val_ds = TensorDataset(X_vl, y_vl)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        wait = 0

        logger.info(f"  TFT regression-only: training (epochs={epochs}, patience={patience})")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                out = self.model(X_b)["regression"].squeeze(-1)
                loss = criterion(out, y_b.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)
            train_loss /= len(train_ds)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out = self.model(X_b)["regression"].squeeze(-1)
                    val_loss += criterion(out, y_b.float()).item() * X_b.size(0)
            val_loss /= len(val_ds) if len(val_ds) > 0 else 1

            scheduler.step(val_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"    Epoch {epoch+1}/{epochs} - train: {train_loss:.6f} - val: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"    Early stopping at epoch {epoch+1}, best val_loss: {best_loss:.6f}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def _fit_tft_multitask(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        y_ter_tr: torch.Tensor, y_ter_vl: torch.Tensor,
        lr: float, batch_size: int, epochs: int, patience: int, weight_decay: float,
    ) -> "DeepModel":
        """Full multi-task TFT training with Kendall 2018 loss."""
        X_tr = torch.FloatTensor(X_train).to(self.device)
        y_reg_tr = torch.FloatTensor(y_train).to(self.device)
        y_bin_tr = torch.FloatTensor(y_train).to(self.device)
        X_vl = torch.FloatTensor(X_val).to(self.device)
        y_reg_vl = torch.FloatTensor(y_val).to(self.device)
        y_bin_vl = torch.FloatTensor(y_val).to(self.device)

        train_ds = TensorDataset(X_tr, y_reg_tr, y_bin_tr, y_ter_tr)
        val_ds = TensorDataset(X_vl, y_reg_vl, y_bin_vl, y_ter_vl)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        criterion = MultiTaskLoss(task=self.task, n_classes=self.n_classes).to(self.device)

        class_weights = None
        if self.n_classes == 2:
            n_pos = float(np.sum(y_train == 1))
            n_neg = float(len(y_train) - n_pos)
            pos_weight = max(n_neg / max(n_pos, 1), 1.0)
            class_weights = torch.where(
                y_bin_tr == 1,
                torch.tensor(pos_weight, device=self.device),
                torch.tensor(1.0, device=self.device),
            )

        best_loss = float("inf")
        best_state = None
        wait = 0

        logger.info(f"  TFT multi-task: training (epochs={epochs}, patience={patience}, ternary=True)")

        for epoch in range(epochs):
            self.model.train()
            criterion.train()
            train_loss = 0
            for X_b, y_reg_b, y_bin_b, y_ter_b in train_loader:
                optimizer.zero_grad()
                out = self.model(X_b)

                loss, losses_dict = criterion(
                    out, y_reg_b, y_bin_b,
                    y_ter_b,
                    class_weights[:y_bin_b.size(0)] if class_weights is not None else None,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)
            train_loss /= len(train_ds)

            self.model.eval()
            criterion.eval()
            val_loss = 0
            with torch.no_grad():
                for X_b, y_reg_b, y_bin_b, y_ter_b in val_loader:
                    out = self.model(X_b)
                    vl, _ = criterion(
                        out, y_reg_b, y_bin_b,
                        y_ter_b,
                        None,
                    )
                    val_loss += vl.item() * X_b.size(0)
            val_loss /= len(val_ds) if len(val_ds) > 0 else 1

            scheduler.step(val_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"    Epoch {epoch+1}/{epochs} - train: {train_loss:.6f} - val: {val_loss:.6f}"
                    f" [reg={losses_dict['loss_reg']:.4f}, bin={losses_dict['loss_binary']:.4f}, ter={losses_dict.get('loss_ternary', 0):.4f}]"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"    Early stopping at epoch {epoch+1}, best val_loss: {best_loss:.6f}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)

        if isinstance(out, dict):
            if self.n_classes > 2:
                return np.argmax(out["ternary"].cpu().numpy(), axis=1).flatten()
            if self.task == "classification" and self.n_classes == 2:
                binary = torch.sigmoid(out["binary"]).cpu().numpy().flatten()
                return (binary > 0.5).astype(int)
            return out["regression"].cpu().numpy().flatten()

        out = out.cpu().numpy()
        if self.n_classes > 2:
            return np.argmax(out, axis=1).flatten()
        if self.task == "classification" and self.n_classes == 2:
            return (out.flatten() > 0.5).astype(int)
        return out.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)

        if isinstance(out, dict):
            if self.n_classes > 2:
                ternary_out = out["ternary"].cpu().numpy()
                exp_out = np.exp(ternary_out - np.max(ternary_out, axis=1, keepdims=True))
                return exp_out / exp_out.sum(axis=1, keepdims=True)
            if self.task == "classification" and self.n_classes == 2:
                return torch.sigmoid(out["binary"]).cpu().numpy().flatten()
            return out["regression"].cpu().numpy().flatten()

        out = out.cpu().numpy()
        if self.n_classes > 2:
            exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
            return exp_out / exp_out.sum(axis=1, keepdims=True)
        if self.task == "classification" and self.n_classes == 2:
            return out.flatten()
        return out.flatten()

    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        """Return regression output specifically (for TFT multi-task)."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
        if isinstance(out, dict):
            return out["regression"].cpu().numpy().flatten()
        return out.cpu().numpy().flatten()

    def predict_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return predictions with MC Dropout uncertainty estimates."""
        X_t = torch.FloatTensor(X).to(self.device)
        result = self.model.predict_with_uncertainty(X_t)
        if self.model_type == "tft" and self.n_classes == 2:
            binary_mean = torch.sigmoid(result["binary_mean"]).cpu().numpy().flatten()
        else:
            binary_mean = result["binary_mean"].cpu().numpy().flatten()
        return {
            "regression_mean": result["regression_mean"].cpu().numpy().flatten(),
            "regression_std": result["regression_std"].cpu().numpy().flatten(),
            "binary_mean": binary_mean,
            "binary_std": result["binary_std"].cpu().numpy().flatten(),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        saved_params = dict(self.params)
        saved_params["input_size"] = self.input_size
        if self.model_type == "tft":
            saved_params["n_static_features"] = self.params.get("n_static_features", 8)
        torch.save({
            "model_type": self.model_type,
            "task": self.task,
            "params": saved_params,
            "n_classes": self.n_classes,
            "state_dict": self.model.state_dict(),
        }, path)
        logger.info(f"  Saved {self.model_type}/{self.task} -> {path}")

    @classmethod
    def load(cls, path: str) -> "DeepModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_classes = checkpoint.get("n_classes", 2)
        model = cls(
            model_type=checkpoint["model_type"],
            input_size=checkpoint["params"].get("input_size", 17),
            task=checkpoint["task"],
            params=checkpoint["params"],
            device=device,
            n_classes=n_classes,
        )
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.to(device)
        return model


def get_all_deep_models(input_size: int, task: str, n_classes: int = 2) -> List[DeepModel]:
    models = []
    for model_type in DEEP_PARAMS:
        params = dict(DEEP_PARAMS[model_type])
        if params.get("input_size") == "auto":
            params["input_size"] = input_size
        models.append(DeepModel(model_type, input_size, task, params, n_classes=n_classes))
    return models
