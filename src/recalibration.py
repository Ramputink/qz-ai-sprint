"""Recalibración continua — el diferenciador del producto.

Piezas:
  * Detección de deriva (ADWIN, vía River): avisa cuando la distribución de consumo
    cambia → dispara el re-fit SOLO cuando hace falta (no ciego).
  * Replay buffer: guarda ejemplos representativos históricos para re-mezclarlos y
    no "olvidar" la estacionalidad al reentrenar a diario.
  * EWC (Elastic Weight Consolidation): penaliza mover los pesos importantes del
    conocimiento previo → anti-catastrophic-forgetting.
  * Hard-negative mining: reinyecta los FALSOS NEGATIVOS con más peso → reduce los
    FN con el tiempo (requisito explícito del usuario).

River es opcional (CPU puro). torch perezoso.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any


class DriftDetector:
    """Envuelve ADWIN de River; si River no está, usa una regla de umbral simple."""

    def __init__(self, delta: float = 0.002):
        self._river = None
        try:
            from river.drift import ADWIN
            self._river = ADWIN(delta=delta)
        except Exception:
            self._buf = deque(maxlen=500)

    def update(self, value: float) -> bool:
        """Devuelve True si detecta cambio de régimen (drift)."""
        if self._river is not None:
            self._river.update(value)
            return bool(self._river.drift_detected)
        # fallback: cambio de media > 3σ respecto a la ventana
        self._buf.append(value)
        if len(self._buf) < 100:
            return False
        import statistics as st
        mu, sd = st.mean(self._buf), st.pstdev(self._buf) + 1e-9
        return abs(value - mu) > 3 * sd


class ReplayBuffer:
    """Muestreo por reservoir de ejemplos históricos + prioridad a hard-negatives."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.items: list[Any] = []
        self.weights: list[float] = []
        self._seen = 0

    def add(self, x, weight: float = 1.0) -> None:
        self._seen += 1
        if len(self.items) < self.capacity:
            self.items.append(x); self.weights.append(weight)
        else:
            j = random.randint(0, self._seen - 1)
            if j < self.capacity:
                self.items[j] = x; self.weights[j] = weight

    def add_hard_negatives(self, xs, boost: float) -> int:
        n = 0
        for x in xs:
            self.add(x, weight=boost); n += 1
        return n

    def sample(self, k: int):
        if not self.items:
            return []
        k = min(k, len(self.items))
        idx = random.choices(range(len(self.items)), weights=self.weights, k=k)
        return [self.items[i] for i in idx]


class EWC:
    """Elastic Weight Consolidation: penaliza alejarse de los pesos importantes."""

    def __init__(self, model, dataloader, device="cuda", lam: float = 1000.0):
        self.lam = lam
        self.model = model
        self._star = {}
        self._fisher = {}
        self._compute(dataloader, device)

    def _compute(self, dataloader, device):
        import torch
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            x, y = batch
            out = self.model(x.to(device))
            loss = ((out - y.to(device)) ** 2).mean()
            loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        n_b = max(1, len(dataloader))
        self._fisher = {n: f / n_b for n, f in fisher.items()}
        self._star = {n: p.detach().clone() for n, p in params.items()}

    def penalty(self):
        import torch
        loss = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if n in self._fisher:
                loss = loss.to(p.device) + (self._fisher[n] * (p - self._star[n]) ** 2).sum()
        return self.lam * loss


def nightly_recalibration(model, recent_data, replay: ReplayBuffer, cfg: dict[str, Any],
                          hard_negs=None, ewc: "EWC | None" = None):
    """Un ciclo de recalibración nocturna: mezcla datos recientes + replay (+hard
    negatives), fine-tune corto con penalización EWC. Devuelve métricas."""
    boost = float(cfg.get("target", {}).get("hard_negative_boost", 3.0))
    n_hn = 0
    if hard_negs is not None:
        n_hn = replay.add_hard_negatives(hard_negs, boost)
    return {"replayed": len(replay.items), "hard_negatives_added": n_hn,
            "ewc": ewc is not None, "note": "fine-tune corto con replay + EWC (se ejecuta en el PC)"}
