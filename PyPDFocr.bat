@echo off
start /b python main.py
timeout /t 3 >nul
start http://127.0.0.1:8038