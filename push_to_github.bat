@echo off
cd /d "%~dp0"
echo ========================================================
echo Pushing UI Dashboard to GitHub...
echo ========================================================
git push -u origin feat/interactive-ui-dashboard
if %errorlevel% neq 0 (
    echo.
    echo ----------------------------------------------------
    echo [Notice] If you see an authentication error or permission denied:
    echo 1. Make sure your GitHub account (Bluebird3406) is added as a Collaborator by the repo owner.
    echo 2. Or sign in with a GitHub Personal Access Token (PAT).
    echo ----------------------------------------------------
) else (
    echo.
    echo ========================================================
    echo SUCCESS! Code and UI pushed to GitHub successfully!
    echo ========================================================
)
pause
