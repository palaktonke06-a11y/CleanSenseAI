@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating CleanSense AI virtual environment...
  py -m venv .venv
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto error
pip install -r requirements.txt
if errorlevel 1 goto error
python -m streamlit run streamlit_app.py
if errorlevel 1 goto error
goto end
:error
echo.
echo CleanSense AI could not start. Read the error above and share it for troubleshooting.
pause
:end
endlocal
