"""Descarga resumible de los datasets del plan.

Principios (el sprint dura dias y el PC puede cortarse):
  * REANUDABLE: cada fichero se baja a `<nombre>.part` y se continua con Range
    HTTP si ya hay bytes; nunca se reempieza de cero tras un corte.
  * IDEMPOTENTE: al terminar se escribe `_DOWNLOAD_OK.json` en la carpeta del
    dataset; una segunda pasada lo salta sin tocar la red.
  * TOLERANTE: si una fuente esta caida, se registra el fallo y se sigue con el
    resto. Solo los datasets no opcionales cuentan como error del sprint.

Salida: `data/raw/<clave>/...` (extraido si era zip/tar) y un resumen devuelto al
orquestador con lo que entro y lo que fallo.
"""
from __future__ import annotations

import json
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

from .registry import DatasetSpec

ProgressCB = Callable[[str, float, str], None]

_CHUNK = 1 << 20          # 1 MiB
_UA = {"User-Agent": "qz-ai-sprint/1.0 (+entrenamiento predictivo)"}
_MARKER = "_DOWNLOAD_OK.json"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _http_download(url: str, dest: Path, cb: ProgressCB, key: str,
                   pct_lo: float = 0.0, pct_hi: float = 100.0) -> Path:
    """Baja `url` a `dest` con reanudacion. Devuelve la ruta final."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    headers = dict(_UA)
    if have:
        headers["Range"] = f"bytes={have}-"

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
        if r.status_code == 416:          # ya estaba completo
            part.replace(dest)
            return dest
        r.raise_for_status()
        resuming = r.status_code == 206
        if have and not resuming:         # el servidor ignoro el Range: reempezamos
            have = 0
            part.unlink(missing_ok=True)
        total = int(r.headers.get("Content-Length", 0)) + (have if resuming else 0)

        mode = "ab" if (have and resuming) else "wb"
        done = have if resuming else 0
        t_last = 0.0
        with open(part, mode) as fh:
            for chunk in r.iter_content(_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - t_last > 2.0:
                    t_last = now
                    frac = (done / total) if total else 0.0
                    cb(key, pct_lo + (pct_hi - pct_lo) * frac,
                       f"{_human(done)}/{_human(total) if total else '?'}")
    part.replace(dest)
    return dest


def _seven_zip() -> str | None:
    """Ruta al binario 7z/7za si esta en el PATH (para .7z y .rar de NASA/Paderborn)."""
    for name in ("7z", "7za", "7zr"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _extract_one(archive: Path, out_dir: Path) -> bool:
    """Extrae UN contenedor. Devuelve True si lo reconocio y extrajo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        return True
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
        return True
    if archive.suffix.lower() in (".7z", ".rar"):
        exe = _seven_zip()
        if exe is None:
            try:
                import py7zr
                with py7zr.SevenZipFile(archive, "r") as z:
                    z.extractall(str(out_dir))
                return True
            except Exception:
                return False
        import subprocess
        r = subprocess.run([exe, "x", "-y", f"-o{out_dir}", str(archive)],
                           capture_output=True, text=True)
        return r.returncode == 0
    return False


def _looks_like_web_page(path: Path) -> bool:
    """Detecta que el servidor devolvio una pagina en vez del fichero.

    Pasa cuando una URL de dataset caduca y el sitio responde 200 con su portada o
    una pantalla de login. Sin esta comprobacion el sprint da por descargado un
    dataset que en realidad es HTML, y el fallo aparece horas despues.
    """
    try:
        head = path.open("rb").read(512).lstrip()[:64].lower()
    except Exception:
        return False
    return head.startswith((b"<!doctype html", b"<html", b"{\"", b"<?xml"))


def _extract(archive: Path, out_dir: Path, cb: ProgressCB, key: str) -> None:
    """Extrae el contenedor descargado y, recursivamente, los contenedores que venga
    dentro (los paquetes NASA anidan un .zip/.7z por subconjunto).

    IMPORTANTE: nunca toca el archivo de origen. Extraerlo de nuevo duplicaria
    gigabytes en disco y borrarlo rompe la reanudacion.
    """
    cb(key, 96.0, f"extrayendo {archive.name}")
    if not _extract_one(archive, out_dir):
        return                                     # no es un contenedor conocido

    for _ in range(3):                             # hasta 3 niveles de anidamiento
        pending = [p for p in out_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in (".zip", ".7z", ".rar")
                   and p.resolve() != archive.resolve()
                   and not p.with_suffix("").exists()]
        if not pending:
            break
        for inner in sorted(pending):
            cb(key, 97.0, f"extrayendo anidado {inner.name}")
            try:
                if _extract_one(inner, inner.with_suffix("")):
                    inner.unlink()                 # el contenido ya esta en disco
            except Exception:
                continue                           # corrupto o protegido: se ignora


def _marker(d: Path) -> Path:
    return d / _MARKER


def _already_done(d: Path) -> bool:
    return _marker(d).exists()


def _write_marker(d: Path, spec: DatasetSpec, extra: dict[str, Any]) -> None:
    payload = {"key": spec.key, "method": spec.method, "kind": spec.kind,
               "source": spec.location, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **extra}
    _marker(d).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --- un dataset por metodo -------------------------------------------------
def _do_http(spec: DatasetSpec, d: Path, cb: ProgressCB) -> dict[str, Any]:
    name = spec.location.split("/")[-1].split("?")[0]
    name = name.replace("+", "_")
    archive = d / name
    if not archive.exists():
        _http_download(spec.location, archive, cb, spec.key, 0.0, 95.0)
    size = archive.stat().st_size
    if spec.extract:
        if _looks_like_web_page(archive):
            archive.unlink(missing_ok=True)        # no dejar basura que parezca valida
            raise RuntimeError(
                f"la URL devolvio una pagina web, no un archivo ({size} bytes). "
                "La fuente ha cambiado o exige registro; actualiza registry.py")
        _extract(archive, d, cb, spec.key)
        if not any(p.is_file() and p != archive for p in d.rglob("*")):
            raise RuntimeError(f"el contenedor {archive.name} no produjo ningun fichero")
    return {"archive": archive.name, "bytes": size}


def _do_http_multi(spec: DatasetSpec, d: Path, cb: ProgressCB) -> dict[str, Any]:
    got, failed = [], []
    n = len(spec.files)
    for i, fn in enumerate(spec.files):
        lo, hi = 100.0 * i / n, 100.0 * (i + 1) / n
        dest = d / fn
        if dest.exists():
            got.append(fn)
            continue
        try:
            _http_download(f"{spec.location}/{fn}", dest, cb, spec.key, lo, hi)
            got.append(fn)
        except Exception as e:                    # un .mat suelto puede faltar
            failed.append({"file": fn, "error": str(e)[:200]})
    if not got:
        raise RuntimeError(f"ningun fichero descargado ({len(failed)} fallos)")
    return {"files_ok": len(got), "files_failed": failed}


def _do_hf(spec: DatasetSpec, d: Path, cb: ProgressCB) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    cb(spec.key, 5.0, f"snapshot Hugging Face {spec.location}")
    path = snapshot_download(repo_id=spec.location, repo_type="dataset",
                             local_dir=str(d), max_workers=8)
    return {"snapshot": str(path)}


def _do_kaggle(spec: DatasetSpec, d: Path, cb: ProgressCB) -> dict[str, Any]:
    import kaggle
    cb(spec.key, 5.0, f"kaggle {spec.location}")
    kaggle.api.dataset_download_files(spec.location, path=str(d), unzip=True)
    return {"kaggle": spec.location}


def _do_manual(spec: DatasetSpec, d: Path, cb: ProgressCB) -> dict[str, Any]:
    raise RuntimeError(
        f"'{spec.key}' requiere descarga manual desde {spec.location} "
        "(registro o formato .rar). Deja los ficheros en " + str(d))


_DISPATCH = {"http": _do_http, "http_multi": _do_http_multi,
             "hf": _do_hf, "kaggle": _do_kaggle, "manual": _do_manual}


# --- API publica -----------------------------------------------------------
def download_all(specs: list[DatasetSpec], data_dir: str | Path,
                 cb: ProgressCB | None = None) -> dict[str, Any]:
    """Descarga todas las specs bajo `<data_dir>/raw/<clave>/`.

    Devuelve un resumen {ok: [...], skipped: [...], failed: [...]}. Solo lanza
    excepcion si falla un dataset NO opcional y no habia nada usable en disco.
    """
    cb = cb or (lambda k, p, m: None)
    root = Path(data_dir) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"ok": [], "skipped": [], "failed": []}

    for spec in specs:
        d = root / spec.key
        d.mkdir(parents=True, exist_ok=True)
        if _already_done(d):
            summary["skipped"].append(spec.key)
            cb(spec.key, 100.0, "ya descargado")
            continue
        cb(spec.key, 0.0, f"iniciando ({spec.gb} GB aprox)")
        try:
            info = _DISPATCH[spec.method](spec, d, cb)
            _write_marker(d, spec, info)
            summary["ok"].append(spec.key)
            cb(spec.key, 100.0, "completo")
        except Exception as e:
            entry = {"key": spec.key, "optional": spec.optional, "error": str(e)[:300]}
            summary["failed"].append(entry)
            cb(spec.key, 100.0, f"FALLO: {str(e)[:120]}")

    fatal = [f for f in summary["failed"] if not f["optional"]]
    if fatal and not summary["ok"] and not summary["skipped"]:
        raise RuntimeError(f"ningun dataset disponible; fallos: {fatal}")
    return summary


def available(data_dir: str | Path) -> list[str]:
    """Claves de dataset que ya estan descargadas y listas en disco."""
    root = Path(data_dir) / "raw"
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob(f"*/{_MARKER}"))


def purge(data_dir: str | Path, key: str) -> None:
    """Borra un dataset (para reintentar una descarga corrupta)."""
    d = Path(data_dir) / "raw" / key
    if d.exists():
        shutil.rmtree(d)
