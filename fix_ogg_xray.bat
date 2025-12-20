@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo OGG Re-encoder for X-Ray Engine
echo Uses ffmpeg (decode) + oggenc (encode)
echo ====================================================
echo.

:: Check if ffmpeg is available
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] ffmpeg not found!
    echo.
    echo Install: winget install ffmpeg
    echo Or download from: https://ffmpeg.org/download.html
    echo.
    pause
    exit /b 1
)

:: Check if oggenc is available
where oggenc >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] oggenc not found!
    echo.
    echo Download from: https://www.rarewares.org/ogg-oggenc.php
    echo Extract oggenc.exe and place in this folder or add to PATH
    echo.
    pause
    exit /b 1
)

:: Count ogg files
set count=0
for %%f in (*.ogg) do (
    echo %%f | findstr /i ".bak" >nul
    if errorlevel 1 (
        set /a count+=1
    )
)

if %count%==0 (
    echo No .ogg files found in current directory.
    echo Place this script in the folder with .ogg files and run again.
    pause
    exit /b 0
)

echo Found %count% .ogg file(s) to process
echo.
echo This will:
echo  - Re-encode all .ogg files with X-Ray compatible headers
echo  - Create backup files (.ogg.bak) of originals
echo.
set /p confirm="Continue? (Y/N): "
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

:: Create temp directory
set "tempdir=%TEMP%\ogg_fix_%RANDOM%"
mkdir "%tempdir%" 2>nul

for %%f in (*.ogg) do (
    :: Skip backup files
    echo %%f | findstr /i ".bak" >nul
    if errorlevel 1 (
        set /a current+=1
        echo [!current!/%count%] %%f
        
        :: Decode to WAV using ffmpeg
        ffmpeg -y -i "%%f" -ar 44100 "%tempdir%\temp.wav" >nul 2>&1
        
        if exist "%tempdir%\temp.wav" (
            :: Re-encode with oggenc (produces X-Ray compatible headers)
            oggenc "%tempdir%\temp.wav" -q 6 -o "%tempdir%\temp.ogg" >nul 2>&1
            
            if exist "%tempdir%\temp.ogg" (
                :: Create backup
                if not exist "%%~nf.ogg.bak" (
                    copy "%%f" "%%~nf.ogg.bak" >nul 2>&1
                )
                
                :: Replace original
                copy /y "%tempdir%\temp.ogg" "%%f" >nul 2>&1
                echo   [OK] Re-encoded
                set /a success+=1
            ) else (
                echo   [ERROR] oggenc failed
                set /a failed+=1
            )
            del "%tempdir%\temp.wav" 2>nul
            del "%tempdir%\temp.ogg" 2>nul
        ) else (
            echo   [ERROR] ffmpeg decode failed
            set /a failed+=1
        )
    )
)

:: Cleanup
rmdir /s /q "%tempdir%" 2>nul

echo --------------------------------------------------
echo.
echo Done! Re-encoded: %success%, Failed: %failed%
echo.
if %success% gtr 0 (
    echo Backups saved as .ogg.bak files
    echo To delete all backups: del *.ogg.bak
)
echo.
pause
