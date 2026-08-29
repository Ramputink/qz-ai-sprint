#!/bin/bash
# Lanza los barridos de OEDI cuando la GPU quede libre.
#
# Exige 3 lecturas consecutivas por debajo del umbral: un unico instante de holgura
# puede ser el hueco entre dos lotes del proceso que la esta ocupando. Espera
# ademas a que OEDI este preprocesado, porque el barrido lo primero que hace es
# cargarlo.
#
# Uso:  bash scripts/esperar_gpu_y_barrer.sh
set -u
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UMBRAL_MIB=${UMBRAL_MIB:-8000}
NPZ="$RAIZ/data/processed/oedi_comstock_consumo.npz"
PY="$RAIZ/.venv/Scripts/python.exe"

libres=0
echo "[$(date +%H:%M:%S)] esperando: GPU < ${UMBRAL_MIB} MiB y OEDI preprocesado"
while true; do
  usada=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ -f "$NPZ" ] && [ "$usada" -lt "$UMBRAL_MIB" ]; then
    libres=$((libres+1))
  else
    libres=0
  fi
  [ "$libres" -ge 3 ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] GPU libre (${usada} MiB) y datos listos: lanzando barridos"
"$PY" -u "$RAIZ/scripts/barridos_oedi.py" 2>&1 | grep -Ev "^\[2026|INFO |Warning"
