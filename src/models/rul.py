"""RUL (Remaining Useful Life) — el núcleo predictivo.

TCN (Temporal Convolutional Network): convoluciones causales dilatadas. Predice
cuántos ciclos/horas de vida le quedan al equipo → de ahí sale la ANTICIPACIÓN de
≥10 días. Barato, paraleliza mejor que LSTM, cuantiza bien para edge.

torch se importa de forma perezosa: este módulo solo se usa en ejecución real (PC).
"""
from __future__ import annotations

from typing import Any


def _torch():
    import torch
    return torch


def build_tcn(n_features: int, channels=(64, 64, 64, 64), kernel: int = 3, dropout: float = 0.1):
    """Construye el TCN. Devuelve un nn.Module."""
    torch = _torch()
    import torch.nn as nn

    class Chomp(nn.Module):
        def __init__(self, s): super().__init__(); self.s = s
        def forward(self, x): return x[:, :, :-self.s].contiguous() if self.s > 0 else x

    class TemporalBlock(nn.Module):
        def __init__(self, ci, co, k, d):
            super().__init__()
            pad = (k - 1) * d
            self.net = nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ci, co, k, padding=pad, dilation=d)), Chomp(pad),
                nn.ReLU(), nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Conv1d(co, co, k, padding=pad, dilation=d)), Chomp(pad),
                nn.ReLU(), nn.Dropout(dropout),
            )
            self.down = nn.Conv1d(ci, co, 1) if ci != co else None
            self.relu = nn.ReLU()
        def forward(self, x):
            out = self.net(x)
            res = x if self.down is None else self.down(x)
            return self.relu(out + res)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ci = n_features
            for i, co in enumerate(channels):
                layers.append(TemporalBlock(ci, co, kernel, 2 ** i))
                ci = co
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(channels[-1], 1)  # RUL escalar
        def forward(self, x):            # x: (B, T, F)
            h = self.tcn(x.transpose(1, 2))          # (B, C, T)
            return self.head(h[:, :, -1]).squeeze(-1)  # (B,)

    return TCN()


def rul_loss(pred, target, fn_weight: float = 1.0):
    """Pérdida asimétrica: penaliza MÁS subestimar el desgaste (predecir más vida
    de la real = falso negativo peligroso). Alinea con 'minimizar falsos negativos'."""
    torch = _torch()
    err = pred - target
    # si pred > target (optimista, FN): penaliza fn_weight×; si pesimista: 1×
    w = torch.where(err > 0, torch.as_tensor(fn_weight, device=err.device), torch.as_tensor(1.0, device=err.device))
    return (w * err.pow(2)).mean()


def make_step_fn(model, optimizer, cfg: dict[str, Any]):
    """Devuelve una función step(batch)->loss que el orquestador llama en bucle."""
    torch = _torch()
    fn_w = float(cfg.get("target", {}).get("fn_weight", 1.0))
    use_bf16 = cfg.get("train", {}).get("precision", "bf16") == "bf16"

    def step(batch):
        x, y = batch
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (use_bf16 and torch.cuda.is_available()) else _nullctx()
        with ctx:
            pred = model(x)
            loss = rul_loss(pred, y, fn_w)
        loss.backward()
        optimizer.step()
        return float(loss.detach())
    return step


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def lead_time_metric(pred_rul, true_rul, threshold_rul: float, hours_per_unit: float = 1.0) -> dict[str, float]:
    """Mide la anticipación: cuánto antes del fallo el modelo cruza el umbral de alarma.
    Devuelve lead_time (en horas y días) y si cumple el objetivo ≥10 días."""
    np = _import_np()
    pred = np.asarray(pred_rul)
    # primer índice donde la RUL predicha baja del umbral (dispara alarma)
    alarm_idx = np.argmax(pred <= threshold_rul) if np.any(pred <= threshold_rul) else len(pred) - 1
    # el fallo real ocurre cuando true_rul llega a 0 (último punto)
    lead_units = (len(pred) - 1) - alarm_idx
    lead_hours = lead_units * hours_per_unit
    return {"lead_units": float(lead_units), "lead_hours": float(lead_hours),
            "lead_days": round(lead_hours / 24.0, 2)}


def _import_np():
    import numpy as np
    return np
