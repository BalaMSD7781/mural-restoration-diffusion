@echo off
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
echo Restores every image in data\samples with the full pipeline (GPU machine) and caches the results.
.venv\Scripts\python.exe -m tools.precompute_demo --steps 30
echo.
echo Done. For a CPU-only machine, copy data\samples\cache into its data\samples folder.
pause
