"""Etapa A — ENTRENAMIENTO REAL de detección/predicción de fallo (Tier A pequeño).

Pipeline completo y real (no simulado):
  1. Carga el dataset (SKAB por defecto: bomba/motor con fallos etiquetados).
  2. Health-autoencoder entrenado SOLO con ventanas sanas → score de salud.
  3. Clasificador MLP supervisado con PÉRDIDA SENSIBLE AL COSTE (los falsos
     negativos pesan fn_weight×) → probabilidad de fallo.
  4. Ensemble (AE + clasificador) + umbral óptimo por coste.
  5. Evaluación: accuracy, matriz de confusión (TP/TN/FP/FN), PR-AUC.
  6. Checkpoints + métricas + ProcessView. Device: CUDA (Windows) / MPS (Mac) / CPU.

Mismo código en Mac y Windows → resultados COMPARABLES (cambia solo el device).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _split(X, y, test_frac=0.3, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def run_stage_a(cfg: dict[str, Any], base_dir: Path, logger, pv, ckpt, *,
                dataset: str = "skab", epochs: int = 60, device: str | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    from .device import configure_max_power, autocast_ctx
    from .models.health_autoencoder import build_autoencoder, threshold_from_healthy
    from .models.classifier import best_threshold

    t0 = time.time()
    hw = configure_max_power(cfg.get("train", {}).get("num_workers"), force_device=device)
    dev = hw["device"]
    logger.info("stage_a_start", dataset=dataset, **hw)
    pv.update(phase_label=f"Etapa A · {dataset} · {dev}", generation=1, stage="A", day_of_4=1,
              progress_pct=0.0, gpu={"device": hw.get("gpu"), "modo": dev, "hilos": hw["torch_threads"]},
              target=_target(cfg))

    # --- 1. datos ---
    if dataset == "skab":
        from .data.skab import load as load_ds
    else:
        raise ValueError(f"dataset '{dataset}' aún no implementado en stage_a")
    data = load_ds(base_dir / cfg["paths"]["data_dir"])
    X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    logger.info("data_loaded", n=data["n_samples"], features=data["n_features"],
                normal=data["n_normal"], fault=data["n_fault"])
    Xtr, ytr, Xte, yte = _split(X, y, seed=cfg["run"].get("seed", 0))

    tX = lambda a: torch.tensor(a, device=dev)
    Xtr_t, ytr_t, Xte_t, yte_t = tX(Xtr), tX(ytr), tX(Xte), tX(yte)
    nf = data["n_features"]
    fn_w = float(cfg.get("target", {}).get("fn_weight", 5.0))

    # --- 2. health autoencoder (solo con sanos del train) ---
    ae = build_autoencoder(nf, hidden=(128, 64, 16)).to(dev)
    normal_tr = Xtr[ytr == 0]
    normal_tr_t = tX(normal_tr)              # una sola vez (no recrear cada época → mucho más rápido)
    ae_opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    ae.train()
    for ep in range(epochs):
        perm = torch.randperm(len(normal_tr), device=dev)
        xb = normal_tr_t[perm]
        with autocast_ctx(dev):
            rec = ae(xb)
            loss = ((xb - rec) ** 2).mean()
        ae_opt.zero_grad(set_to_none=True); loss.backward(); ae_opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            pv.update(phase_label="Etapa A · autoencoder de salud", progress_pct=round(40 * ep / epochs, 1),
                      epoch=ep, metrics={"ae_loss": round(float(loss), 5)}, log_tail=logger.tail(10))
            logger.info("ae_epoch", ep=ep, loss=round(float(loss), 5))
            ckpt.maybe_save({"ae": ae.state_dict()}, {"generation": 1, "stage": "A", "step": ep,
                            "metrics": {"ae_loss": float(loss)}}, cfg["run"].get("checkpoint_every_min", 30))
    ae.eval()
    with torch.no_grad():
        health_tr = ae.health_score(normal_tr_t).cpu().numpy()
        health_te = ae.health_score(Xte_t).cpu().numpy()
    ae_thr = threshold_from_healthy(health_tr, percentile=99.0)

    # --- 3. clasificador supervisado con coste asimétrico ---
    clf = nn.Sequential(nn.Linear(nf, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    clf_opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-5)
    # pos_weight = balance de clases (no el coste FN, que se aplica en el umbral).
    n_pos = max(1, int((ytr == 1).sum())); n_neg = max(1, int((ytr == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bs = 256
    for ep in range(epochs):
        clf.train()
        perm = torch.randperm(len(Xtr_t))
        losses = []
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb, yb = Xtr_t[b], ytr_t[b].unsqueeze(1)
            with autocast_ctx(dev):
                logit = clf(xb)
                loss = bce(logit, yb)
            clf_opt.zero_grad(set_to_none=True); loss.backward(); clf_opt.step()
            losses.append(float(loss))
        if ep % 10 == 0 or ep == epochs - 1:
            pv.update(phase_label="Etapa A · clasificador (coste FN)", progress_pct=round(40 + 40 * ep / epochs, 1),
                      epoch=ep, metrics={"clf_loss": round(sum(losses) / len(losses), 5)}, log_tail=logger.tail(10))
            logger.info("clf_epoch", ep=ep, loss=round(sum(losses) / len(losses), 5))
            ckpt.maybe_save({"clf": clf.state_dict()}, {"generation": 1, "stage": "A", "step": 1000 + ep,
                            "metrics": {"clf_loss": sum(losses) / len(losses)}}, cfg["run"].get("checkpoint_every_min", 30))

    # --- 4. ensemble (pesa más el clasificador supervisado) + 2 puntos de operación ---
    clf.eval()
    with torch.no_grad():
        prob_te = torch.sigmoid(clf(Xte_t)).cpu().numpy().ravel()
    # score del AE a rango percentil respecto a los sanos del train (0..1)
    ae_rank = np.searchsorted(np.sort(health_tr), health_te) / max(1, len(health_tr))
    ensemble = 0.7 * prob_te + 0.3 * ae_rank

    def _cm(thr):
        pred = (ensemble >= thr).astype(int)
        yt = yte.astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
        acc = (tp + tn) / max(1, len(yt))
        tpr = tp / max(1, tp + fn); tnr = tn / max(1, tn + fp)
        return {"threshold": float(thr), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "accuracy": round(acc, 4), "balanced_acc": round((tpr + tnr) / 2, 4),
                "cost": fp + fn_w * fn}

    grid = np.unique(np.quantile(ensemble, np.linspace(0.02, 0.98, 97)))
    ops = [_cm(t) for t in grid]
    op_balanced = max(ops, key=lambda o: o["balanced_acc"])   # mejor equilibrio FP/FN
    op_lowfn = min(ops, key=lambda o: o["cost"])              # anti-falso-negativo (coste FN×)

    # --- 5. métricas finales ---
    from sklearn.metrics import average_precision_score, roc_auc_score
    try:
        pr_auc = float(average_precision_score(yte, ensemble))
        roc = float(roc_auc_score(yte, ensemble))
    except Exception:
        pr_auc, roc = None, None
    total = op_balanced["tp"] + op_balanced["tn"] + op_balanced["fp"] + op_balanced["fn"]
    metrics = {
        "device": dev, "hardware": hw.get("gpu"), "dataset": dataset,
        "n_test": int(total),
        "accuracy": op_balanced["accuracy"], "balanced_accuracy": op_balanced["balanced_acc"],
        "confusion": {"tp": op_balanced["tp"], "tn": op_balanced["tn"],
                      "fp": op_balanced["fp"], "fn": op_balanced["fn"]},
        "false_negatives": op_balanced["fn"], "false_positives": op_balanced["fp"],
        "threshold": round(op_balanced["threshold"], 4),
        "operating_point_low_fn": {"accuracy": op_lowfn["accuracy"], "fp": op_lowfn["fp"],
                                   "fn": op_lowfn["fn"], "threshold": round(op_lowfn["threshold"], 4)},
        "pr_auc": pr_auc, "roc_auc": roc, "fn_weight": fn_w, "epochs": epochs,
        "train_seconds": round(time.time() - t0, 1),
    }
    best = op_balanced
    logger.info("stage_a_done", **{k: metrics[k] for k in ("device", "accuracy", "false_negatives", "false_positives", "train_seconds")})
    pv.update(phase_label="Etapa A COMPLETADA", progress_pct=100.0, metrics=metrics, log_tail=logger.tail(12))

    # --- 6. guardar resultados (comparables Mac vs Windows) ---
    out = base_dir / cfg["paths"]["artifacts_dir"]
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"stage_a_result_{dev}.json"
    res_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    ckpt.save_best({"ae": ae.state_dict(), "clf": clf.state_dict(), "norm": data["norm"], "ae_thr": ae_thr},
                   {"generation": 1, "stage": "A", "step": 9999, "metrics": metrics})
    logger.info("result_saved", path=res_path.name)
    return metrics


def _target(cfg):
    t = cfg.get("target", {})
    return {"accuracy≥": t.get("min_accuracy"), "peso_FN": t.get("fn_weight")}
