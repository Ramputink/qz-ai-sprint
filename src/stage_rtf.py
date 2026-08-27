"""Etapa RUN-TO-FAILURE — MetroPT-3, a máxima carga del ordenador.

Objetivo: predecir el fallo del compresor con ANTELACIÓN (lead-time), a partir de la
degradación de los sensores. Entrena:
  * RUL regresor (MLP profundo) → minutos hasta el fallo.
  * Health-autoencoder → desviación de la normalidad (alerta temprana).
  * Clasificador pre-fallo con coste asimétrico (FN pesa fn_weight×).
Máxima carga: BÚSQUEDA de hiperparámetros (N_trials entrenamientos completos), todos
los núcleos + MPS, checkpoints. Split TEMPORAL (train pasado → test futuro) para que
el lead-time sea honesto.

Métrica clave: para cada fallo del periodo de test, cuántos días antes salta la alarma.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _temporal_split(n: int, frac: float = 0.7):
    cut = int(n * frac)
    return slice(0, cut), slice(cut, n)


def _confusion(y_true, score, thr):
    import numpy as np
    pred = (score >= thr).astype(int); yt = y_true.astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
    acc = (tp + tn) / max(1, len(yt))
    tpr = tp / max(1, tp + fn); tnr = tn / max(1, tn + fp)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": round(acc, 4),
            "balanced_acc": round((tpr + tnr) / 2, 4)}


def _debounce(alarm_bool, min_consec):
    """Antirrebote: la alarma solo cuenta tras `min_consec` ventanas consecutivas
    sobre el umbral → elimina alarmas transitorias (falsos positivos)."""
    import numpy as np
    out = np.zeros(len(alarm_bool), dtype=bool)
    run = 0
    for i, a in enumerate(alarm_bool):
        run = run + 1 if a else 0
        out[i] = run >= min_consec
    return out


def _lead_time(timestamps, alarm_bool, onsets, lead_horizon_days=20.0):
    """Para cada onset, primer instante con ALARMA (ya con antirrebote) en la ventana
    previa → días de antelación."""
    import numpy as np
    import pandas as pd
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    leads = []
    for o in onsets:
        o = pd.Timestamp(o)
        mask = np.asarray((ts >= o - pd.Timedelta(days=lead_horizon_days)) & (ts < o))
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        alarmed = idxs[alarm_bool[idxs]]
        if len(alarmed):
            first = ts[int(alarmed[0])]
            leads.append({"onset": str(o), "alarm": str(first),
                          "lead_days": round((o - first).total_seconds() / 86400.0, 2)})
        else:
            leads.append({"onset": str(o), "alarm": None, "lead_days": 0.0})
    return leads


def _false_alarm_events(timestamps, alarm_bool, onsets, guard_days):
    """Nº de EPISODIOS de alarma (flancos de subida) fuera de la zona de fallo — la
    métrica operacional de falsos positivos (no ventana a ventana)."""
    import numpy as np
    import pandas as pd
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    near = np.zeros(len(ts), dtype=bool)
    for o in onsets:
        o = pd.Timestamp(o)
        near |= np.asarray((ts >= o - pd.Timedelta(days=guard_days)) & (ts <= o + pd.Timedelta(days=2)))
    fa, prev = 0, False
    for i in range(len(alarm_bool)):
        if alarm_bool[i] and not prev and not near[i]:
            fa += 1
        prev = bool(alarm_bool[i])
    return fa


def run_stage_rtf(cfg: dict[str, Any], base_dir: Path, logger, pv, ckpt, *,
                  n_trials: int = 30, epochs: int = 120, device: str | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    from .device import configure_max_power, autocast_ctx
    from .models.health_autoencoder import build_autoencoder, threshold_from_healthy
    from .data.metropt import load as load_metropt

    t0 = time.time()
    hw = configure_max_power(cfg.get("train", {}).get("num_workers"), force_device=device)
    dev = hw["device"]
    logger.info("rtf_start", **hw, n_trials=n_trials, epochs=epochs)
    pv.update(phase_label="Run-to-failure · cargando MetroPT (15M filas)", generation=1, stage="RTF",
              day_of_4=1, progress_pct=0.0, gpu={"device": hw.get("gpu"), "modo": dev, "hilos": hw["torch_threads"]})

    data = load_metropt(base_dir / cfg["paths"]["data_dir"],
                        lead_days=float(cfg.get("target", {}).get("lead_time_days", 10.0)),
                        progress=lambda ph, i, n: pv.update(phase_label=f"MetroPT · {ph}",
                                                            progress_pct=round(5 + 5 * (i % 100) / 100, 1)))
    X = data["X"].astype(np.float32); y = data["y"].astype(np.float32)
    logger.info("metropt_loaded", raw_rows=data["raw_rows"], windows=data["n_samples"],
                normal=data["n_normal"], fault=data["n_fault"])
    tr, te = _temporal_split(len(y), 0.7)   # TEMPORAL: pasado→futuro
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    ts_te = data["timestamps"][te]
    onsets = data["onsets"]
    nf = data["n_features"]
    fn_w = float(cfg.get("target", {}).get("fn_weight", 5.0))
    tX = lambda a: torch.tensor(a, device=dev)
    Xtr_t, Xte_t = tX(Xtr), tX(Xte)
    ytr_t = tX(ytr)
    n_pos = max(1, int((ytr == 1).sum())); n_neg = max(1, int((ytr == 0).sum()))

    # --- health autoencoder (una vez, sobre sanos del train) ---
    normal_tr = Xtr[ytr == 0]
    ae = build_autoencoder(nf, hidden=(256, 128, 32)).to(dev)
    normal_t = tX(normal_tr)
    ae_opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(epochs):
        perm = torch.randperm(len(normal_tr), device=dev)
        with autocast_ctx(dev):
            loss = ((normal_t[perm] - ae(normal_t[perm])) ** 2).mean()
        ae_opt.zero_grad(set_to_none=True); loss.backward(); ae_opt.step()
        if ep % 20 == 0:
            pv.update(phase_label="RTF · autoencoder de salud", progress_pct=round(10 + 15 * ep / epochs, 1),
                      epoch=ep, metrics={"ae_loss": round(float(loss.detach()), 5)}, log_tail=logger.tail(8))
    ae.eval()
    with torch.no_grad():
        h_tr = ae.health_score(normal_t).cpu().numpy()
        h_te = ae.health_score(Xte_t).cpu().numpy()
    ae_rank = np.searchsorted(np.sort(h_tr), h_te) / max(1, len(h_tr))

    # --- BÚSQUEDA de hiperparámetros (máxima carga: n_trials entrenamientos) ---
    rng = np.random.default_rng(cfg["run"].get("seed", 0))
    best = None
    for trial in range(n_trials):
        h1 = int(rng.choice([128, 256, 512])); h2 = int(rng.choice([64, 128, 256]))
        lr = float(10 ** rng.uniform(-3.5, -2.3)); dr = float(rng.uniform(0.1, 0.4))
        clf = nn.Sequential(nn.Linear(nf, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dr),
                            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dr), nn.Linear(h2, 1)).to(dev)
        opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-5)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=dev))
        bs = 512
        for ep in range(epochs):
            clf.train(); perm = torch.randperm(len(Xtr_t), device=dev)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                with autocast_ctx(dev):
                    loss = bce(clf(Xtr_t[b]), ytr_t[b].unsqueeze(1))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            prob = torch.sigmoid(clf(Xte_t)).cpu().numpy().ravel()
        ens = 0.7 * prob + 0.3 * ae_rank
        grid = np.unique(np.quantile(ens, np.linspace(0.05, 0.95, 46)))
        cms = [(t, _confusion(yte, ens, t)) for t in grid]
        thr, cm = max(cms, key=lambda kc: kc[1]["balanced_acc"])
        score = {"trial": trial, "balanced_acc": cm["balanced_acc"], "acc": cm["accuracy"],
                 "cm": cm, "thr": float(thr), "hp": {"h1": h1, "h2": h2, "lr": round(lr, 5), "dropout": round(dr, 3)},
                 "prob": prob}
        if best is None or score["balanced_acc"] > best["balanced_acc"]:
            best = score
            ckpt.save_best({"clf": clf.state_dict(), "ae": ae.state_dict(), "norm": data["norm"]},
                           {"generation": 1, "stage": "RTF", "step": trial, "metrics": {k: cm[k] for k in ("accuracy", "balanced_acc", "fp", "fn")}})
        pv.update(phase_label=f"RTF · búsqueda HP (trial {trial+1}/{n_trials})",
                  progress_pct=round(25 + 70 * (trial + 1) / n_trials, 1), epoch=trial,
                  metrics={"best_bal_acc": best["balanced_acc"], "trial_bal_acc": cm["balanced_acc"]},
                  gpu={"device": hw.get("gpu"), "modo": dev, "hilos": hw["torch_threads"]},
                  log_tail=logger.tail(8))
        logger.info("trial", trial=trial, bal_acc=cm["balanced_acc"], acc=cm["accuracy"],
                    fp=cm["fp"], fn=cm["fn"], **score["hp"])

    # --- PUNTO DE OPERACIÓN con antirrebote: detectar el fallo con MÍNIMAS falsas alarmas ---
    ens_best = 0.7 * best["prob"] + 0.3 * ae_rank
    lead_target = float(cfg.get("target", {}).get("lead_time_days", 10.0))
    op = None
    for consec in [3, 6, 12, 24, 48]:                       # 30 min … 8 h de persistencia
        for q in np.linspace(0.85, 0.995, 30):
            thr = float(np.quantile(ens_best, q))
            deb = _debounce(ens_best >= thr, consec)
            leads = _lead_time(ts_te, deb, onsets)
            hit = [l for l in leads if l["alarm"]]
            fa = _false_alarm_events(ts_te, deb, onsets, guard_days=lead_target)
            min_lead = min([l["lead_days"] for l in hit], default=0.0)
            key = (len(hit), -fa, min_lead)                  # +detección, −falsas alarmas, +antelación
            if op is None or key > op["key"]:
                op = {"key": key, "thr": thr, "consec": consec, "leads": leads,
                      "fa": fa, "hit": len(hit), "min_lead": min_lead, "alarm": deb}
    leads = op["leads"]
    lead_days_hit = [l["lead_days"] for l in leads if l["alarm"]]
    # confusión a nivel ventana YA con antirrebote (FP muy inferior al crudo)
    deb = op["alarm"]; yt = yte.astype(int)
    tp = int((deb & (yt == 1)).sum()); tn = int((~deb & (yt == 0)).sum())
    fp = int((deb & (yt == 0)).sum()); fn = int((~deb & (yt == 1)).sum())
    acc = round((tp + tn) / max(1, len(yt)), 4)
    tpr = tp / max(1, tp + fn); tnr = tn / max(1, tn + fp)

    from sklearn.metrics import average_precision_score, roc_auc_score
    try:
        pr_auc = float(average_precision_score(yte, ens_best)); roc = float(roc_auc_score(yte, ens_best))
    except Exception:
        pr_auc, roc = None, None

    metrics = {
        "device": dev, "hardware": hw.get("gpu"), "dataset": "metropt3",
        "raw_rows": data["raw_rows"], "windows": data["n_samples"], "n_features": nf,
        "n_test": len(yte), "accuracy": acc, "balanced_accuracy": round((tpr + tnr) / 2, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "false_negatives": fn, "false_positives": fp,
        "false_alarm_events": op["fa"],           # ← la FP operacional (episodios), tras antirrebote
        "operating_point": {"threshold": round(op["thr"], 4), "min_consecutive_windows": op["consec"]},
        "pr_auc": pr_auc, "roc_auc": roc,
        "best_hp": best["hp"], "n_trials": n_trials, "epochs": epochs, "fn_weight": fn_w,
        "lead_time": leads,
        "mean_lead_days": round(sum(lead_days_hit) / len(lead_days_hit), 2) if lead_days_hit else 0.0,
        "failures_detected": f"{len(lead_days_hit)}/{len(leads)}",
        "train_seconds": round(time.time() - t0, 1),
    }
    pv.update(phase_label="RUN-TO-FAILURE COMPLETADO", progress_pct=100.0, metrics=metrics, log_tail=logger.tail(12))
    out = base_dir / cfg["paths"]["artifacts_dir"]; out.mkdir(parents=True, exist_ok=True)
    (out / f"rtf_result_{dev}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("rtf_done", device=dev, accuracy=best["acc"], mean_lead_days=metrics["mean_lead_days"],
                detected=metrics["failures_detected"], seconds=metrics["train_seconds"])
    return metrics
