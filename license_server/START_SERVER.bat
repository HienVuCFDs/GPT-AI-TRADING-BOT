@echo off
title License Server
cd /d "%~dp0"
echo Starting License Server...
echo.
python run_server.py
pause
