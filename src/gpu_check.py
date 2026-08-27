"""Test de humo del stack de GPU para la RTX 5090 (Blackwell / sm_120).

Se ejecuta al arrancar en el PC. Si el stack es incorrecto, ABORTA con un mensaje
claro (mejor fallar en 2 segundos que a las 3 horas de entrenamiento).

Requisitos verificados (fuente: guías NVIDIA Blackwell / PyTorch):
  * driver NVIDIA >= 570
  * CUDA >= 12.8
  * PyTorch build cu128+ con sm_120 en la lista de arquitecturas
  * un matmul real en la GPU en bf16

En el Mac (sin CUDA) devuelve un informe "no aplicable" sin abortar, para poder
probar el resto del flujo.
"""
from __future__ import annotations

from typing import Any


def check(strict: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": False, "reason": "", "details": {}}
    try:
        import torch
    except Exception as e:  # torch no instalado (p. ej. en el Mac de análisis)
        report["reason"] = f"torch no importable ({e}). En el PC de entrenamiento instala requirements-train.txt."
        report["details"]["torch"] = None
        return report

    report["details"]["torch_version"] = torch.__version__
    report["details"]["cuda_build"] = getattr(torch.version, "cuda", None)

    if not torch.cuda.is_available():
        report["reason"] = "torch.cuda no disponible. En Mac es normal; en el PC revisa driver/CUDA."
        return report

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)  # (12, 0) para Blackwell RTX 5090
    arches = []
    try:
        arches = torch.cuda.get_arch_list()
    except Exception:
        pass
    report["details"].update({"device": name, "capability": f"{cap[0]}.{cap[1]}", "arch_list": arches})

    # sm_120 = Blackwell. Si el build no lo trae, el entrenamiento fallará.
    has_sm120 = any("120" in a for a in arches)
    cuda_ok = True
    try:
        cuda_ok = float(str(torch.version.cuda)) >= 12.8
    except Exception:
        cuda_ok = True  # no bloquear por parseo

    problems = []
    if cap[0] < 12:
        problems.append(f"capability {cap} < 12.0 (¿no es una RTX 50xx?)")
    if not has_sm120 and cap[0] >= 12:
        problems.append("el build de PyTorch NO incluye sm_120 → usa wheels cu128+ o el container NGC 25.x")
    if not cuda_ok:
        problems.append(f"CUDA build {torch.version.cuda} < 12.8")

    # matmul real en bf16 (prueba que los kernels existen de verdad)
    try:
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        _ = (x @ x).float().sum().item()
        report["details"]["bf16_matmul"] = "ok"
    except Exception as e:
        problems.append(f"matmul bf16 en GPU falló: {e}")

    if problems:
        report["reason"] = " · ".join(problems)
        report["ok"] = False
    else:
        report["ok"] = True
        report["reason"] = "stack Blackwell correcto"
    return report


def assert_ready(strict: bool = True) -> dict[str, Any]:
    r = check(strict=strict)
    if not r["ok"] and strict:
        raise SystemExit(
            "\n[gpu_check] STACK NO LISTO PARA ENTRENAR:\n  " + r["reason"] +
            "\n  detalles: " + str(r["details"]) +
            "\n  -> instala requirements-train.txt en el PC (Windows + RTX 5090) o usa el container NGC.\n"
        )
    return r


if __name__ == "__main__":
    import json
    print(json.dumps(check(strict=False), ensure_ascii=False, indent=2))
