@echo off
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
echo Runs the full pipeline once on a test image. First run downloads ~4 GB (SD inpainting + ControlNet).
.venv\Scripts\python.exe -m tools.smoke_test
echo.
echo Open smoke_full.png. Check the output above if anything failed.
pause
