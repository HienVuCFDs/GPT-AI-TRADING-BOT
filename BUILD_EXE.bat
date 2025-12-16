@echo off
title Building Trading Bot Executable
color 0A

echo ============================================================
echo    TRADING BOT - BUILD TO EXECUTABLE
echo ============================================================
echo.
echo This will create TradingBot.exe in the 'dist' folder
echo Build time: approximately 10-30 minutes (first time)
echo.
echo Press any key to start building...
pause >nul

echo.
echo [1/4] Cleaning old build files...
rd /s /q build 2>nul
rd /s /q dist 2>nul
del /q *.spec 2>nul
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo.
echo [2/4] Starting PyInstaller build...
echo This may take 10-30 minutes. Please wait...
echo.

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name=TradingBot ^
    --clean ^
    --noconfirm ^
    --hidden-import=PyQt5 ^
    --hidden-import=PyQt5.QtCore ^
    --hidden-import=PyQt5.QtGui ^
    --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=PyQt5.QtNetwork ^
    --hidden-import=MetaTrader5 ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=requests ^
    --hidden-import=sklearn ^
    --hidden-import=sklearn.ensemble ^
    --hidden-import=sklearn.preprocessing ^
    --hidden-import=ta ^
    --hidden-import=xgboost ^
    --hidden-import=torch ^
    --hidden-import=sqlite3 ^
    --hidden-import=json ^
    --hidden-import=threading ^
    --hidden-import=multiprocessing ^
    --hidden-import=dotenv ^
    --hidden-import=pytz ^
    --hidden-import=dateutil ^
    --add-data="ai_server_config.json;." ^
    --add-data="ai_trading_config.json;." ^
    --add-data="notification_config.json;." ^
    --add-data="smart_entry_config.json;." ^
    --add-data="license_config.json;." ^
    --add-data="trailing_stop_config.json;." ^
    --add-data="mt5_data_config.ini;." ^
    app.py

if exist "dist\TradingBot.exe" (
    echo.
    echo ============================================================
    echo    BUILD SUCCESSFUL!
    echo ============================================================
    echo.
    echo [3/4] Copying config files to dist folder...
    copy /Y "*.json" "dist\" 2>nul
    copy /Y "*.ini" "dist\" 2>nul
    copy /Y "users_db.json" "dist\" 2>nul
    
    echo.
    echo [4/4] Creating data folders in dist...
    if not exist "dist\logs" mkdir "dist\logs"
    if not exist "dist\data" mkdir "dist\data"
    
    echo.
    echo ============================================================
    echo    DONE!
    echo ============================================================
    echo.
    echo Your executable is at: dist\TradingBot.exe
    echo.
    echo IMPORTANT:
    echo  - Copy the 'data' folder to 'dist' if needed
    echo  - All config files (.json) have been copied to 'dist'
    echo  - First run may take longer ^(extracting files^)
    echo.
    
    :: Get file size
    for %%A in ("dist\TradingBot.exe") do set size=%%~zA
    set /a sizeMB=%size% / 1048576
    echo File size: approximately %sizeMB% MB
    echo.
    
) else (
    echo.
    echo ============================================================
    echo    BUILD FAILED!
    echo ============================================================
    echo.
    echo Check the error messages above.
    echo Common issues:
    echo  - Missing dependencies: pip install -r requirements.txt
    echo  - Antivirus blocking: Add exception for this folder
    echo  - Disk space: Need at least 2GB free space
    echo.
)

echo.
echo Press any key to exit...
pause >nul
