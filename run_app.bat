@echo off
cd /d "%~dp0"
echo Launching Dynamic Concept Explorer UI...
uv run streamlit run src/dynamic_concept_explorer/app.py
pause
