"""Logging estructurado en JSONL + consola.

Cada línea de log es un objeto JSON (una por evento) → fácil de leer y graficar en
el Mac sin dependencias raras. También imprime en consola en formato humano para
seguir el proceso en el PC durante los 4 días.

Sin dependencias de terceros (solo stdlib) para que corra en cualquier entorno.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """Logger de la ejecución. Escribe a `<logs_dir>/run.jsonl` y a consola."""

    def __init__(self, logs_dir: str | Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.logs_dir / "run.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, level: str, event: str, **fields: Any) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            "level": level,
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        # consola humana
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        stream = sys.stderr if level in ("error", "warn") else sys.stdout
        print(f"[{rec['ts']}] {level.upper():5} {event} {extra}", file=stream, flush=True)

    def info(self, event: str, **f: Any) -> None:
        self.log("info", event, **f)

    def warn(self, event: str, **f: Any) -> None:
        self.log("warn", event, **f)

    def error(self, event: str, **f: Any) -> None:
        self.log("error", event, **f)

    def tail(self, n: int = 20) -> list[str]:
        """Últimas n líneas de log (para el ProcessView)."""
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        return [ln.rstrip("\n") for ln in lines[-n:]]

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
