@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo OGG Re-encoder for X-Ray Engine (RECURSIVE)
echo Uses ffmpeg + oggenc - Processes ALL subdirectories
echo ====================================================
echo.

:: Check tools
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] ffmpeg not found!
    echo Install: winget install ffmpeg
    pause
    exit /b 1
)

where oggenc >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] oggenc not found!
    echo Download from: https://www.rarewares.org/ogg-oggenc.php
    pause
    exit /b 1
)

:: Count files recursively
set count=0
for /r %%f in (*.ogg) do (
    echo %%f | findstr /i ".bak" >nul
    if errorlevel 1 (
        set /a count+=1
    )
)

if %count%==0 (
    echo No .ogg files found.
    pause
    exit /b 0
)

echo Found %count% .ogg file(s) in all subdirectories
echo Working directory: %cd%
echo.
set /p confirm="Re-encode all for X-Ray compatibility? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Processing...
echo --------------------------------------------------

set success=0
set failed=0
set current=0
set "tempdir=%TEMP%\ogg_fix_%RANDOM%"
mkdir "%tempdir%" 2>nul

for /r %%f in (*.ogg) do (
    echo %%f | findstr /i ".bak" >nul
    if errorlevel 1 (
        set /a current+=1
        echo [!current!/%count%] %%~nxf
        
        ffmpeg -y -i "%%f" -ar 44100 "%tempdir%\temp.wav" >nul 2>&1
        
        if exist "%tempdir%\temp.wav" (
            oggenc "%tempdir%\temp.wav" -q 6 -o "%tempdir%\temp.ogg" >nul 2>&1
            
            if exist "%tempdir%\temp.ogg" (
                if not exist "%%~dpnf.ogg.bak" (
                    copy "%%f" "%%~dpnf.ogg.bak" >nul 2>&1
                )
                copy /y "%tempdir%\temp.ogg" "%%f" >nul 2>&1
                echo   [OK]
                set /a success+=1
            ) else (
                echo   [ERROR] oggenc
                set /a failed+=1
            )
            del "%tempdir%\temp.wav" 2>nul
            del "%tempdir%\temp.ogg" 2>nul
        ) else (
            echo   [ERROR] ffmpeg
            set /a failed+=1
        )
    )
)

rmdir /s /q "%tempdir%" 2>nul

echo --------------------------------------------------
echo Done! Re-encoded: %success%, Failed: %failed%
echo.
if %success% gtr 0 (
    echo Backups: .ogg.bak files
    echo Delete all backups: del /s *.ogg.bak
)
pause
