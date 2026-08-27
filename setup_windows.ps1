# setup_windows.ps1 — instalación en el PC de entrenamiento (Windows + RTX 5090).
#
# Crea un entorno virtual, instala PyTorch para Blackwell (cu128) y el resto de
# dependencias, y verifica que la GPU está lista (sm_120). Ejecutar en PowerShell:
#
#     powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#
# Requisitos previos (una vez): driver NVIDIA >= 570 y CUDA >= 12.8 instalados.
# Comprobar con: nvidia-smi  (debe decir "CUDA Version: 12.8" o superior).

$ErrorActionPreference = "Stop"
Write-Host "== QuantumZIGMA sprint · setup Windows (RTX 5090 / Blackwell) ==" -ForegroundColor Magenta

# 1) Python
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { Write-Error "Python no encontrado. Instala Python 3.11/3.12 y reintenta."; exit 1 }
python --version

# 2) venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creando entorno virtual .venv ..." -ForegroundColor Cyan
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3) PyTorch para Blackwell (cu128). Si en 2026 hay un canal más nuevo (cu129/cu130),
#    cámbialo aquí. NO instalar torch desde requirements: necesita este índice.
Write-Host "Instalando PyTorch (cu128) para la RTX 5090 ..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4) Resto de dependencias
Write-Host "Instalando dependencias de entrenamiento ..." -ForegroundColor Cyan
pip install -r requirements-train.txt

# 5) Verificación del stack (aborta si la GPU no está lista)
Write-Host "Verificando el stack de GPU ..." -ForegroundColor Cyan
python run.py --gpu-check

Write-Host ""
Write-Host "== LISTO ==" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANTE: en cada terminal NUEVA, activa el entorno ANTES de ejecutar:" -ForegroundColor Yellow
Write-Host "    .\.venv\Scripts\Activate.ps1        (el prompt debe empezar por (.venv))" -ForegroundColor Yellow
Write-Host ""
Write-Host "Luego:"
Write-Host "Prueba el flujo sin entrenar:   python run.py --dry-run"
Write-Host "Lanza el sprint de 4 dias:      python run.py"
Write-Host "Retomar tras un corte:          python run.py --resume"
Write-Host "Panel en vivo: abre processview\index.html en el navegador."
