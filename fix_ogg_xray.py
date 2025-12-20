#!/usr/bin/env python3
"""
OGG Re-encoder for X-Ray Engine

Uses ffmpeg (decode) + oggenc (encode) to re-encode .ogg files 
with X-Ray compatible Vorbis headers.

Requirements: 
  - ffmpeg (for decoding)
  - oggenc from vorbis-tools (for encoding with compatible headers)
  
Install:
  - Windows ffmpeg: winget install ffmpeg
  - Windows oggenc: https://www.rarewares.org/ogg-oggenc.php
  - Linux: sudo apt install ffmpeg vorbis-tools
  - macOS: brew install ffmpeg vorbis-tools

Usage:
    python fix_ogg_xray.py              - Process current directory
    python fix_ogg_xray.py <folder>     - Process specified folder
    python fix_ogg_xray.py -r           - Process recursively
    python fix_ogg_xray.py -r <folder>  - Process folder recursively
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def check_tools():
    """Check if required tools are available"""
    has_ffmpeg = False
    has_oggenc = False
    
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        has_ffmpeg = True
    except FileNotFoundError:
        pass
    
    try:
        subprocess.run(["oggenc", "--version"], capture_output=True, check=False)
        has_oggenc = True
    except FileNotFoundError:
        pass
    
    return has_ffmpeg, has_oggenc

def reencode_ogg(input_path, backup=True):
    """Re-encode a single ogg file"""
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"  [ERROR] File not found")
        return False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "temp.wav"
        temp_ogg = Path(tmpdir) / "temp.ogg"
        
        try:
            # Decode to WAV using ffmpeg
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path), "-ar", "44100", str(wav_path)],
                capture_output=True, text=True
            )
            
            if not wav_path.exists():
                print(f"  [ERROR] ffmpeg decode failed")
                return False
            
            # Re-encode with oggenc (produces X-Ray compatible headers)
            result = subprocess.run(
                ["oggenc", str(wav_path), "-q", "6", "-o", str(temp_ogg)],
                capture_output=True, text=True
            )
            
            if not temp_ogg.exists():
                print(f"  [ERROR] oggenc failed")
                return False
            
            # Create backup
            if backup:
                backup_path = input_path.with_suffix('.ogg.bak')
                if not backup_path.exists():
                    try:
                        shutil.copy2(input_path, backup_path)
                    except:
                        pass
            
            # Replace original
            shutil.copy2(temp_ogg, input_path)
            return True
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False

def process_directory(directory, recursive=False):
    """Process all .ogg files in a directory"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    if recursive:
        ogg_files = list(directory.rglob("*.ogg"))
    else:
        ogg_files = list(directory.glob("*.ogg"))
    
    ogg_files = [f for f in ogg_files if '.bak' not in str(f)]
    
    if not ogg_files:
        print(f"No .ogg files found in {directory}")
        return
    
    print(f"Found {len(ogg_files)} .ogg file(s) to process")
    print("-" * 50)
    
    success = 0
    failed = 0
    
    for i, ogg_file in enumerate(ogg_files, 1):
        rel_path = ogg_file.relative_to(directory) if recursive else ogg_file.name
        print(f"[{i}/{len(ogg_files)}] {rel_path}")
        
        if reencode_ogg(ogg_file):
            print(f"  [OK] Re-encoded")
            success += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Done! Re-encoded: {success}, Failed: {failed}")
    
    if success > 0:
        print(f"\nBackups saved as .ogg.bak files")

def main():
    print("=" * 50)
    print("OGG Re-encoder for X-Ray Engine")
    print("Uses ffmpeg + oggenc")
    print("=" * 50)
    
    has_ffmpeg, has_oggenc = check_tools()
    
    if not has_ffmpeg:
        print("\n[ERROR] ffmpeg not found!")
        print("  Windows: winget install ffmpeg")
        print("  Linux:   sudo apt install ffmpeg")
        print("  macOS:   brew install ffmpeg")
    
    if not has_oggenc:
        print("\n[ERROR] oggenc not found!")
        print("  Windows: https://www.rarewares.org/ogg-oggenc.php")
        print("  Linux:   sudo apt install vorbis-tools")
        print("  macOS:   brew install vorbis-tools")
    
    if not has_ffmpeg or not has_oggenc:
        sys.exit(1)
    
    recursive = "-r" in sys.argv
    args = [a for a in sys.argv[1:] if a != "-r"]
    
    directory = args[0] if args else "."
    
    print(f"\nTarget: {os.path.abspath(directory)}")
    print(f"Recursive: {recursive}\n")
    
    process_directory(directory, recursive)

if __name__ == "__main__":
    main()
