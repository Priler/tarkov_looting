#!/usr/bin/env python3
"""
OGG Re-encoder for X-Ray Engine Compatibility
Fixes "invalid ogg-comment version" errors by re-encoding .ogg files
with stripped metadata and compatible Vorbis settings.

Requirements: ffmpeg must be installed and in PATH

Usage:
    python fix_ogg_files.py              - Process current directory
    python fix_ogg_files.py <folder>     - Process specified folder
    python fix_ogg_files.py -r           - Process current directory recursively
    python fix_ogg_files.py -r <folder>  - Process specified folder recursively
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def reencode_ogg(input_path, backup=True):
    """Re-encode a single ogg file with X-Ray compatible settings"""
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"  [ERROR] File not found: {input_path}")
        return False
    
    # Create temp output path
    temp_path = input_path.with_suffix('.tmp.ogg')
    backup_path = input_path.with_suffix('.ogg.bak')
    
    try:
        # Re-encode with ffmpeg
        # -map_metadata -1 strips all metadata (fixes the version issue)
        # -c:a libvorbis uses Vorbis codec
        # -q:a 6 is good quality (0-10 scale)
        # -ar 44100 ensures standard sample rate
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:a", "libvorbis",
            "-q:a", "6",
            "-map_metadata", "-1",  # Strip all metadata
            "-ar", "44100",         # Standard sample rate
            str(temp_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed: {result.stderr[:200]}")
            if temp_path.exists():
                temp_path.unlink()
            return False
        
        # Create backup if requested
        if backup:
            if backup_path.exists():
                backup_path.unlink()
            shutil.copy2(input_path, backup_path)
        
        # Replace original with re-encoded version
        input_path.unlink()
        temp_path.rename(input_path)
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def process_directory(directory, recursive=False):
    """Process all .ogg files in a directory"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    # Find all .ogg files
    if recursive:
        ogg_files = list(directory.rglob("*.ogg"))
    else:
        ogg_files = list(directory.glob("*.ogg"))
    
    # Filter out backup files and temp files
    ogg_files = [f for f in ogg_files if not f.name.endswith('.bak') and '.tmp.' not in f.name]
    
    if not ogg_files:
        print(f"No .ogg files found in {directory}")
        return
    
    print(f"Found {len(ogg_files)} .ogg file(s) to process")
    print("-" * 50)
    
    success = 0
    failed = 0
    
    for i, ogg_file in enumerate(ogg_files, 1):
        rel_path = ogg_file.relative_to(directory) if recursive else ogg_file.name
        print(f"[{i}/{len(ogg_files)}] Processing: {rel_path}")
        
        if reencode_ogg(ogg_file):
            print(f"  [OK] Re-encoded successfully")
            success += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"Complete! Success: {success}, Failed: {failed}")
    
    if success > 0:
        print(f"\nBackup files created with .ogg.bak extension")
        print("Delete them manually after confirming everything works.")

def main():
    print("=" * 50)
    print("OGG Re-encoder for X-Ray Engine")
    print("=" * 50)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("\n[ERROR] ffmpeg not found!")
        print("Please install ffmpeg and add it to your PATH")
        print("\nDownload: https://ffmpeg.org/download.html")
        print("Or use: winget install ffmpeg")
        sys.exit(1)
    
    # Parse arguments
    recursive = "-r" in sys.argv
    args = [a for a in sys.argv[1:] if a != "-r"]
    
    if args:
        directory = args[0]
    else:
        directory = "."
    
    print(f"\nTarget directory: {os.path.abspath(directory)}")
    print(f"Recursive: {recursive}")
    print()
    
    process_directory(directory, recursive)

if __name__ == "__main__":
    main()
