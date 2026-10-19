@echo off
title AnkiForge — AI Flashcard Generator
cd /d "C:\Users\kabee\anki-forge"

echo ========================================================
echo               Starting AnkiForge...
echo ========================================================
echo.

call "C:\Users\kabee\miniconda3\Scripts\activate.bat" "C:\Users\kabee\miniconda3"
streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo AnkiForge exited with error code %ERRORLEVEL%.
    echo ========================================================
    pause
)
