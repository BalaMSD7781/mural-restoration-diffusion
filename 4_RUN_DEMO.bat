@echo off
cd /d "%~dp0"
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
echo Starting the demo at http://localhost:8501  (close this window to stop)
.venv\Scripts\python.exe -m streamlit run app.py --browser.gatherUsageStats false
pause
