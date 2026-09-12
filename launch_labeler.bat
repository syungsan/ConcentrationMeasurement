@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHONW=%ROOT%WPy64-312101\python\pythonw.exe"
set "LAUNCHER=%ROOT%launcher.py"

if not exist "%PYTHONW%" (
    echo WinPython was not found:
    echo %PYTHONW%
    pause
    exit /b 1
)

if not exist "%LAUNCHER%" (
    echo launcher.py was not found:
    echo %LAUNCHER%
    pause
    exit /b 1
)

start "Concentration Labeler" /D "%ROOT%" "%PYTHONW%" "%LAUNCHER%"
exit /b 0
