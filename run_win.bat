@echo off

REM Add directories to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;C:\dev\aim\nas-fedot;C:\dev\aim\nas-fedot\cases\mnist

REM Run the Python script
C:\dev\aim\nas-fedot\venv\Scripts\python.exe C:/dev/aim/nas-fedot/cases\mnist\mnist_classification.py
