"""Health-index por autoencoder — alerta temprana de fallo + anomalía de consumo.

Se entrena SOLO con datos "sanos": aprende a reconstruir la normalidad. El error de
reconstrucción creciente = el equipo se está degradando (alerta antes de que falle)
o hay consumo anómalo. El mismo AE, con features de tráfico Modbus/OPC-UA, sirve de
IDS OT (bloque de intrusión) → sinergia de coste.

torch perezoso.
"""
from __future__ import annotations


def _torch():
    import torch
    return torch


def build_autoencoder(n_features: int, hidden=(128, 64, 16)):
    torch = _torch()
    import torch.nn as nn

    def enc_block(i, o):
        return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [n_features, *hidden]
            enc = []
            for i in range(len(dims) - 1):
                enc += enc_block(dims[i], dims[i + 1])
            self.encoder = nn.Sequential(*enc)
            dec = []
            rdims = list(reversed(dims))
            for i in range(len(rdims) - 1):
                dec += [nn.Linear(rdims[i], rdims[i + 1])]
                if i < len(rdims) - 2:
                    dec += [nn.BatchNorm1d(rdims[i + 1]), nn.ReLU()]
            self.decoder = nn.Sequential(*dec)
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
        def health_score(self, x):
            """Error de reconstrucción normalizado = índice de salud (0 sano, ↑ degradado)."""
            rec = self.forward(x)
            return ((x - rec) ** 2).mean(dim=1)

    return AE()


def reconstruction_loss(model, x):
    rec = model(x)
    return ((x - rec) ** 2).mean()


def threshold_from_healthy(scores, percentile: float = 99.0) -> float:
    """Umbral de alarma a partir de la distribución de errores en datos sanos."""
    import numpy as np
    return float(np.percentile(np.asarray(scores), percentile))
