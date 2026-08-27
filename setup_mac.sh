#!/usr/bin/env bash
# setup_mac.sh — entorno de SOLO ANÁLISIS en el Mac (CPU, sin CUDA).
# Para revisar el progreso que llega del PC: abrir paquetes, cargar checkpoints, graficar.
set -euo pipefail

echo "== QuantumZIGMA sprint · setup Mac (solo análisis) =="
PYTHON=${PYTHON:-python3}
$PYTHON --version

if [ ! -d ".venv-mac" ]; then
  echo "Creando entorno virtual .venv-mac ..."
  $PYTHON -m venv .venv-mac
fi
# shellcheck disable=SC1091
source .venv-mac/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-analyze.txt

echo ""
echo "== LISTO (Mac) =="
echo "Ver el progreso de un paquete recibido:  python analyze/view_progress.py artifacts/gen1_etapa1a_*.zip --open"
echo "Cargar un checkpoint:                    python analyze/load_checkpoint.py checkpoints/latest.json"
echo "Graficar métricas:                       python analyze/plots.py logs/run.jsonl"
echo "Probar el flujo completo (simulado):     python run.py --dry-run"
