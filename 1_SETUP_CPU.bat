@echo off
cd /d "%~dp0"
echo ===== SETUP (CPU laptop) =====
py -3.11 -V >nul 2>&1 && (set PY=py -3.11) || (set PY=python)
echo Using: %PY%
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv || goto :err
)
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :err
echo --- PyTorch (CPU build) ---
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :err
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err
if not exist models\attention_unet.keras (
  echo.
  echo !! models\attention_unet.keras is missing - place the trained Attention U-Net there (exact name: attention_unet.keras).
)
echo.
echo Setup finished.
pause
exit /b 0
:err
echo.
echo !! Setup failed. Scroll up to the first red error message.
pause
exit /b 1
