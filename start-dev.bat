@echo off
echo Starting AI Crop Advisor Development Environment...
echo.

echo Starting Flask Backend...
start "Flask Backend" cmd /k "cd /d %~dp0 && python app.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting React Frontend...
start "React Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit...
pause > nul
