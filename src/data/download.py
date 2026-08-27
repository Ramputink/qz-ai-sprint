"""Descarga de datasets según el catálogo (registry.py).

Soporta 4 métodos: hf, kaggle, url, manual. Reanudable (HTTP Range para 'url'),
con progreso que se refleja en el ProcessView y en el log. Los métodos que
necesitan librerías (huggingface_hub, kaggle) las importan de forma perezosa, así
que `--list` y el resto del paquete funcionan en el Mac sin tenerlas instaladas.

CLI:
  python -m src.data.download --list                 # lista el plan y tamaños
  python -m src.data.download --keys metropt3 skab   # descarga esos datasets
  python -m src.data.download --config config.yaml   # descarga los del config
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

try:  # permite ejecutar como módulo o directamente
    from .registry import DatasetSpec, resolve, total_gb, REGISTRY
except ImportError:  # pragma: no cover
    from registry import DatasetSpec, resolve, total_gb, REGISTRY  # type: ignore


ProgressCB = Optional[Callable[[str, float, str], None]]  # (dataset_key, pct, msg)


def _report(cb: ProgressCB, key: str, pct: float, msg: str) -> None:
    if cb:
        cb(key, pct, msg)
    else:
        print(f"[download] {key} {pct:5.1f}% {msg}", flush=True)


def download_url(spec: DatasetSpec, dest: Path, cb: ProgressCB = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = spec.files or [spec.location]
    for i, url in enumerate(files):
        fname = url.split("/")[-1] or f"{spec.key}_{i}.bin"
        out = dest / fname
        resume_from = out.stat().st_size if out.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "qz-ai-sprint/1.0"})
        if resume_from:
            req.add_header("Range", f"bytes={resume_from}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0)) + resume_from
                mode = "ab" if resume_from else "wb"
                done = resume_from
                with open(out, mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)  # 1 MB
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        pct = (done / total * 100) if total else 0.0
                        _report(cb, spec.key, pct, f"{done/1e6:.0f}/{(total or 0)/1e6:.0f} MB {fname}")
        except Exception as e:
            _report(cb, spec.key, 0.0, f"ERROR url {fname}: {e}")
            raise


def download_hf(spec: DatasetSpec, dest: Path, cb: ProgressCB = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        _report(cb, spec.key, 0.0, f"instala 'huggingface_hub' en el PC ({e})")
        raise
    _report(cb, spec.key, 1.0, f"snapshot_download {spec.location} …")
    snapshot_download(repo_id=spec.location, repo_type="dataset",
                      local_dir=str(dest / spec.key), resume_download=True)
    _report(cb, spec.key, 100.0, "completado (HF)")


def download_kaggle(spec: DatasetSpec, dest: Path, cb: ProgressCB = None) -> None:
    try:
        import kaggle  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        _report(cb, spec.key, 0.0, f"instala 'kaggle' y ~/.kaggle/kaggle.json en el PC ({e})")
        raise
    api = KaggleApi()
    api.authenticate()
    out = dest / spec.key
    out.mkdir(parents=True, exist_ok=True)
    _report(cb, spec.key, 1.0, f"kaggle download {spec.location} …")
    api.dataset_download_files(spec.location, path=str(out), unzip=True, quiet=False)
    _report(cb, spec.key, 100.0, "completado (Kaggle)")


def download_manual(spec: DatasetSpec, dest: Path, cb: ProgressCB = None) -> None:
    _report(cb, spec.key, 0.0,
            f"MANUAL: requiere registro. Descárgalo de {spec.location} y colócalo en {dest/spec.key}/ (se salta)")


_DISPATCH = {"url": download_url, "hf": download_hf, "kaggle": download_kaggle, "manual": download_manual}


def download_all(specs: list[DatasetSpec], data_dir: Path, cb: ProgressCB = None) -> list[str]:
    ok: list[str] = []
    for spec in specs:
        _report(cb, spec.key, 0.0, f"→ {spec.name} ({spec.method}, ~{spec.gb} GB)")
        fn = _DISPATCH.get(spec.method, download_manual)
        try:
            fn(spec, data_dir, cb)
            ok.append(spec.key)
        except Exception as e:
            _report(cb, spec.key, 0.0, f"fallo (continúo con el resto): {e}")
    return ok


def _keys_from_config(cfg_path: str) -> list[str]:
    import yaml  # lazy
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    keys: list[str] = []
    for _, lst in (cfg.get("datasets") or {}).items():
        keys.extend(lst or [])
    return keys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Descarga de datasets del sprint QuantumZIGMA")
    ap.add_argument("--list", action="store_true", help="lista el plan y sale")
    ap.add_argument("--keys", nargs="*", help="claves de dataset a descargar")
    ap.add_argument("--config", help="config.yaml para tomar los datasets")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args(argv)

    if a.config and not a.keys:
        keys = _keys_from_config(a.config)
    elif a.keys:
        keys = a.keys
    else:
        keys = list(REGISTRY.keys())

    specs = resolve(keys)
    if a.list:
        for s in specs:
            print(f"  {s.key:24} {s.method:6} ~{s.gb} GB  {s.location[:70]}")
        print(f"\nTotal plan: ~{total_gb(specs)} GB · {len(specs)} datasets")
        return 0

    ok = download_all(specs, Path(a.data_dir))
    print(f"\nDescargados {len(ok)}/{len(specs)}: {', '.join(ok)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
