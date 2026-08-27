"""Selección de dispositivo y máxima potencia — mismo código en Windows y Mac.

Prioridad de dispositivo: CUDA (RTX 5090 en Windows) → MPS (Apple Silicon en Mac) → CPU.
Configura el máximo de hilos para exprimir la CPU (AMD 9950X3D / Apple M2 Pro) y
activa las optimizaciones de matmul.

torch perezoso: no se importa hasta que se llama.
"""
from __future__ import annotations

import os
from typing import Any


def pick_device(prefer: str | None = None) -> str:
    import torch
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_max_power(num_threads: int | None = None, force_device: str | None = None) -> dict[str, Any]:
    """Ajusta hilos y flags para rendimiento máximo. Devuelve un informe.
    `force_device` respeta el dispositivo elegido por el usuario (--device)."""
    import torch
    cores = os.cpu_count() or 4
    threads = num_threads or cores
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(2, threads // 2))
    except Exception:
        pass
    # BLAS/OMP también al máximo (afecta a numpy/scipy/sklearn)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(threads))

    dev = force_device or pick_device()
    info: dict[str, Any] = {"device": dev, "cpu_cores": cores, "torch_threads": threads}

    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        info["gpu"] = torch.cuda.get_device_name(0)
        info["amp_dtype"] = "bfloat16"
    elif dev == "mps":
        # MPS: bf16 no siempre; usamos fp32 estable (M2 Pro va sobrado en modelos pequeños)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        info["gpu"] = "Apple MPS (Metal)"
        info["amp_dtype"] = "float32"
    else:
        info["gpu"] = None
        info["amp_dtype"] = "float32"
    return info


def autocast_ctx(device: str):
    """Contexto de precisión mixta según dispositivo (bf16 en CUDA; fp32 en MPS/CPU)."""
    import torch
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    class _null:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _null()


if __name__ == "__main__":
    import json
    try:
        print(json.dumps(configure_max_power(), ensure_ascii=False, indent=2))
    except ImportError:
        print("torch no instalado en este intérprete (usa el venv de entrenamiento).")
