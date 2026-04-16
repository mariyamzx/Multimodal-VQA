@echo off
call .venv\Scripts\activate
python -u main.py > app.log 2>&1
pause
