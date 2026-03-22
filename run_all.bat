@echo off
echo ========================================================
echo       Hybrid HR Analytics System - One-Click Run
echo ========================================================

:: 1. Seed Database (Create Users automatically)
echo.
echo [1/3] Checking Database and Seeding Data...
cd backend
call venv\Scripts\activate
python seed_data.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Database seeding failed! 
    echo Please ensure your PostgreSQL server is running on localhost:5432.
    echo.
    pause
)
cd ..

:: 2. Start Backend
echo.
echo [2/3] Starting Backend Server...
start "Backend - FastAPI" cmd /k "cd backend && call venv\Scripts\activate && uvicorn main:app --reload"

:: 3. Start Frontend
echo.
echo [3/3] Starting Frontend Server...
start "Frontend - React" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo SYSTEM LAUNCHED!
echo.
echo Backend API:  http://localhost:8000
echo Frontend UI:  http://localhost:5173
echo.
echo NOTE: Ensure PostgreSQL and MongoDB are running.
echo ========================================================
pause
