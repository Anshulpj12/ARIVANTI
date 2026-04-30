@echo off
title SP 21:2005 RAG System
color 0B
echo.
echo  ====================================================
echo   SP 21:2005 - Building Materials Intelligence System
echo  ====================================================
echo.

cd /d "%~dp0"

:: Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [WARN] No venv found. Using system Python.
)

:: Check/build FAISS index
echo [1/2] Checking FAISS index...
if not exist "backend\index_store\faiss.index" (
    echo       Index not found. Running ingestion pipeline...
    echo       This may take 2-5 minutes on first run.
    echo.
    python backend\ingest.py
    if errorlevel 1 (
        echo [ERROR] Ingestion failed.
        pause
        exit /b 1
    )
) else (
    echo       Index found. Skipping ingestion.
)

:: Start server
echo [2/2] Starting server on http://localhost:8000
echo.
echo  ====================================================
echo   Open your browser: http://localhost:8000
echo   Press Ctrl+C to stop the server.
echo  ====================================================
echo.

:: Open browser
start http://localhost:8000

:: Run server
cd backend
python server.py
