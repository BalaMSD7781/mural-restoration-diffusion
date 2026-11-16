@echo off
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
.venv\Scripts\python.exe -m unittest discover -s tests
pause
