@echo off
REM setup_windows.bat — doble clic para instalar en Windows (llama al .ps1).
REM Instala PyTorch cu128 (RTX 5090), dependencias y verifica la GPU.
echo == QuantumZIGMA sprint · instalador Windows ==
powershell -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
echo.
echo Si hubo errores arriba, revisa que nvidia-smi muestre CUDA 12.8+ y driver 570+.
pause
