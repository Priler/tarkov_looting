#!/usr/bin/env python3
"""
GIF/APNG/Image Sequence to STALKER DDS Frames Converter
--------------------------------------------------------
Converts animated images (GIF, APNG) or image sequences to DDS frames
for use with S.T.A.L.K.E.R. Anomaly mods.

Two modes:
  - search (default): 256x256 square frames for item loading spinners
  - indicator: 256x32 rectangular frames for HUD loading bars

Usage:
    python gif_to_stalker_dds.py input.gif [options]
    python gif_to_stalker_dds.py input.gif --mode indicator --seq
    python gif_to_stalker_dds.py ./icons --batch --mode indicator --seq
    
Requirements:
    pip install Pillow

Examples:
    # Search spinners (256x256):
    python gif_to_stalker_dds.py spinner.gif --seq
    python gif_to_stalker_dds.py spinner.gif --no-crop --fill 80 --seq
    
    # HUD indicators (256x32):
    python gif_to_stalker_dds.py loading_bar.gif --mode indicator --seq
    python gif_to_stalker_dds.py ./bars --batch --mode indicator --seq
"""

import os
import sys
import struct
import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# Mode configurations
MODE_CONFIG = {
    'search': {
        'width': 256,
        'height': 256,
        'prefix': 'icon_looting_',
        'seq_path': 'ui\\loot_searching',
        'description': 'Item loading spinner (256x256 square)'
    },
    'indicator': {
        'width': 256,
        'height': 32,
        'prefix': 'indicator_',
        'seq_path': 'ui\\loot_indicator',
        'description': 'HUD loading bar (256x32 rectangular)'
    }
}


def write_dds_header(width, height):
    """Generate DDS header for uncompressed ARGB8888 format."""
    # DDS magic number
    magic = b'DDS '
    
    # DDS_HEADER structure (124 bytes)
    dwSize = 124
    dwFlags = 0x1 | 0x2 | 0x4 | 0x1000  # CAPS | HEIGHT | WIDTH | PIXELFORMAT
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = width * 4  # 4 bytes per pixel (ARGB)
    dwDepth = 0
    dwMipMapCount = 0
    dwReserved1 = bytes(44)  # 11 DWORDs reserved
    
    # DDS_PIXELFORMAT structure (32 bytes)
    pf_dwSize = 32
    pf_dwFlags = 0x1 | 0x40  # ALPHAPIXELS | RGB
    pf_dwFourCC = 0
    pf_dwRGBBitCount = 32
    pf_dwRBitMask = 0x00FF0000  # Red
    pf_dwGBitMask = 0x0000FF00  # Green
    pf_dwBBitMask = 0x000000FF  # Blue
    pf_dwABitMask = 0xFF000000  # Alpha
    
    # DDS_HEADER caps
    dwCaps = 0x1000  # TEXTURE
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0
    
    header = magic
    header += struct.pack('<I', dwSize)
    header += struct.pack('<I', dwFlags)
    header += struct.pack('<I', dwHeight)
    header += struct.pack('<I', dwWidth)
    header += struct.pack('<I', dwPitchOrLinearSize)
    header += struct.pack('<I', dwDepth)
    header += struct.pack('<I', dwMipMapCount)
    header += dwReserved1
    
    # Pixel format
    header += struct.pack('<I', pf_dwSize)
    header += struct.pack('<I', pf_dwFlags)
    header += struct.pack('<I', pf_dwFourCC)
    header += struct.pack('<I', pf_dwRGBBitCount)
    header += struct.pack('<I', pf_dwRBitMask)
    header += struct.pack('<I', pf_dwGBitMask)
    header += struct.pack('<I', pf_dwBBitMask)
    header += struct.pack('<I', pf_dwABitMask)
    
    header += struct.pack('<I', dwCaps)
    header += struct.pack('<I', dwCaps2)
    header += struct.pack('<I', dwCaps3)
    header += struct.pack('<I', dwCaps4)
    header += struct.pack('<I', dwReserved2)
    
    return header


def get_content_bbox(img):
    """Get bounding box of non-transparent content."""
    if img.mode != 'RGBA':
        return (0, 0, img.width, img.height)
    
    # Get alpha channel
    alpha = img.split()[3]
    bbox = alpha.getbbox()
    
    if bbox is None:
        return (0, 0, img.width, img.height)
    
    return bbox


def normalize_frames(frames):
    """
    Normalize all frames to prevent jumping animation.
    
    This extracts the content from each frame, finds the maximum content size,
    then centers each frame's content on a canvas of that size.
    """
    if not frames:
        return frames
    
    # Convert all frames to RGBA and extract content
    contents = []
    max_width = 0
    max_height = 0
    
    for img in frames:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        bbox = get_content_bbox(img)
        if bbox:
            content = img.crop(bbox)
            contents.append(content)
            max_width = max(max_width, content.width)
            max_height = max(max_height, content.height)
        else:
            contents.append(img)
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
    
    if max_width == 0 or max_height == 0:
        return frames
    
    # Center each content on a canvas of max size
    normalized = []
    for content in contents:
        # Create transparent canvas of max size
        canvas = Image.new('RGBA', (max_width, max_height), (0, 0, 0, 0))
        
        # Calculate position to center content
        x = (max_width - content.width) // 2
        y = (max_height - content.height) // 2
        
        # Paste content centered
        canvas.paste(content, (x, y), content)
        normalized.append(canvas)
    
    return normalized


def image_to_dds(img, output_path, width, height, fill_percent=None, no_crop=False):
    """Convert PIL Image to DDS file (ARGB8888 uncompressed)."""
    # Ensure RGBA mode
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    if no_crop:
        # Don't crop - scale entire frame as-is (good for rotating animations)
        if fill_percent:
            # Scale entire frame to fill percentage of canvas
            target_w = int(width * fill_percent / 100)
            target_h = int(height * fill_percent / 100)
            scale = min(target_w / img.width, target_h / img.height)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            # Fit in canvas without scaling up
            if img.width > width or img.height > height:
                # Scale down to fit
                scale = min(width / img.width, height / img.height)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    elif fill_percent:
        # Crop to content bounding box first, then scale
        bbox = get_content_bbox(img)
        img = img.crop(bbox)
        
        # Calculate target size based on fill percentage
        target_w = int(width * fill_percent / 100)
        target_h = int(height * fill_percent / 100)
        
        # Scale to fill target size (maintain aspect ratio)
        scale = min(target_w / img.width, target_h / img.height)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        # Original behavior: resize to fit within canvas
        scale = min(width / img.width, height / img.height)
        if scale < 1:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with transparent background
    new_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Center the resized image
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    new_img.paste(img, (x, y), img if img.mode == 'RGBA' else None)
    
    # Get pixel data as BGRA (DDS uses BGRA order for "ARGB8888")
    pixels = new_img.load()
    pixel_data = bytearray()
    
    for py in range(height):
        for px in range(width):
            r, g, b, a = pixels[px, py]
            # DDS ARGB8888 is actually stored as BGRA in memory
            pixel_data.extend([b, g, r, a])
    
    # Write DDS file
    with open(output_path, 'wb') as f:
        f.write(write_dds_header(width, height))
        f.write(pixel_data)


def extract_frames_from_gif(gif_path):
    """Extract all frames from a GIF or APNG."""
    frames = []
    img = Image.open(gif_path)
    
    try:
        while True:
            # Convert frame to RGBA
            frame = img.convert('RGBA')
            frames.append(frame.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    
    return frames


def load_image_sequence(folder_path):
    """Load images from a folder as a sequence."""
    folder = Path(folder_path)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.webp'}
    
    # Find all image files
    image_files = sorted([
        f for f in folder.iterdir() 
        if f.suffix.lower() in extensions
    ])
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    frames = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGBA')
        frames.append(img)
    
    return frames


def redistribute_frames(frames, target_count):
    """Redistribute frames to match target count using interpolation."""
    if len(frames) == target_count:
        return frames
    
    result = []
    for i in range(target_count):
        # Map target index to source index
        src_idx = i * (len(frames) - 1) / (target_count - 1) if target_count > 1 else 0
        src_idx = min(int(round(src_idx)), len(frames) - 1)
        result.append(frames[src_idx])
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Convert GIF/APNG/images to STALKER DDS frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  search (default)  - 256x256 square frames for item loading spinners
                      Textures go in: ui/loot_searching/{folder_name}/
  indicator         - 256x32 rectangular frames for HUD loading bars
                      Textures go in: ui/loot_indicator/{folder_name}/

Examples:
  # Search spinners (256x256 square):
  %(prog)s spinner.gif --seq                    # Basic conversion
  %(prog)s spinner.gif --no-crop --fill 80 --seq  # For rotating animations
  %(prog)s ./icons --batch --no-crop --seq      # Batch process folder
  
  # HUD indicators (256x32 rectangular):
  %(prog)s loading.gif --mode indicator --seq
  %(prog)s ./bars --batch --mode indicator --seq
  
  # Custom sizes (overrides mode defaults):
  %(prog)s anim.gif --width 128 --height 128 --seq
        """
    )
    
    parser.add_argument('input', help='Input GIF/APNG file or folder with images')
    parser.add_argument('-o', '--output', default=None, help='Output folder (default: same name as input file)')
    parser.add_argument('-m', '--mode', choices=['search', 'indicator'], default='search',
                        help='Conversion mode: search (256x256) or indicator (256x32)')
    parser.add_argument('-f', '--frames', type=int, default=13, help='Number of output frames (default: 13)')
    parser.add_argument('-p', '--prefix', default=None, help='Filename prefix (default: based on mode)')
    parser.add_argument('-W', '--width', type=int, default=None, help='Output width in pixels (default: based on mode)')
    parser.add_argument('-H', '--height', type=int, default=None, help='Output height in pixels (default: based on mode)')
    parser.add_argument('-F', '--fill', type=int, default=None, metavar='PERCENT',
                        help='Scale content to fill PERCENT of canvas (e.g., 80 = 80%%). Crops empty space first.')
    parser.add_argument('-n', '--normalize', action='store_true',
                        help='Normalize frames by centering content (for pulsing/scaling animations)')
    parser.add_argument('--no-crop', action='store_true',
                        help='Keep original frame intact - don\'t crop to content (for rotating/spinning animations)')
    parser.add_argument('--suffix', default=None, help='Filename suffix before extension (default: "b" for search, "" for indicator)')
    parser.add_argument('--start', type=int, default=1, help='Starting frame number (default: 1)')
    parser.add_argument('--seq', action='store_true', help='Also generate .seq file for STALKER animation')
    parser.add_argument('--batch', action='store_true', 
                        help='Process all .png/.gif/.webp files in input folder, creating a subfolder for each')
    
    args = parser.parse_args()
    
    # Get mode configuration
    config = MODE_CONFIG[args.mode]
    
    # Apply mode defaults (can be overridden by explicit args)
    if args.width is None:
        args.width = config['width']
    if args.height is None:
        args.height = config['height']
    if args.prefix is None:
        args.prefix = config['prefix']
    if args.suffix is None:
        args.suffix = 'b' if args.mode == 'search' else ''
    
    args.seq_path = config['seq_path']
    
    input_path = Path(args.input)
    
    print(f"Mode: {args.mode} - {config['description']}")
    print(f"Output size: {args.width}x{args.height}")
    print()
    
    # Batch mode: process all image files in the directory
    if args.batch:
        if not input_path.is_dir():
            print(f"Error: --batch requires a directory path, got: {input_path}")
            return
        
        # Find all image files
        image_extensions = {'.gif', '.png', '.webp', '.apng'}
        image_files = [f for f in input_path.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {input_path}")
            return
        
        print(f"Batch mode: Found {len(image_files)} image files in {input_path}")
        print("=" * 60)
        
        for img_file in sorted(image_files):
            output_folder = input_path / img_file.stem
            print(f"\nProcessing: {img_file.name} -> {output_folder.name}/")
            try:
                process_single_image(img_file, output_folder, args)
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n" + "=" * 60)
        print(f"Batch complete! Processed {len(image_files)} images.")
        return
    
    # Single file/folder mode
    # Default output folder is named after input file (without extension)
    if args.output:
        output_folder = Path(args.output)
    else:
        # Use input filename without extension as folder name
        output_folder = Path(input_path.stem)
    
    process_single_image(input_path, output_folder, args)


def process_single_image(input_path, output_folder, args):
    """Process a single image/animation file or folder of frames."""
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load frames
    print(f"  Loading: {input_path.name}")
    
    if input_path.is_dir():
        frames = load_image_sequence(input_path)
        print(f"    Found {len(frames)} images in folder")
    elif input_path.suffix.lower() in {'.gif', '.apng', '.png', '.webp'}:
        try:
            frames = extract_frames_from_gif(input_path)
            print(f"    Extracted {len(frames)} frames from animation")
        except Exception as e:
            # Single image
            img = Image.open(input_path).convert('RGBA')
            frames = [img]
            print(f"    Loaded as single image")
    else:
        # Try to open as single image
        img = Image.open(input_path).convert('RGBA')
        frames = [img]
        print(f"    Loaded as single image")
    
    # Redistribute frames to target count
    if len(frames) != args.frames:
        print(f"    Redistributing {len(frames)} frames to {args.frames} frames")
        frames = redistribute_frames(frames, args.frames)
    
    # Normalize frames to prevent jumping animation
    # Skip if --no-crop is used (they are incompatible for rotating animations)
    if args.normalize and args.no_crop:
        print(f"    Warning: --normalize is ignored when --no-crop is used")
    elif args.normalize:
        print(f"    Normalizing frames...")
        frames = normalize_frames(frames)
    
    # Convert and save
    output_files = []
    for i, frame in enumerate(frames):
        frame_num = args.start + i
        filename = f"{args.prefix}{frame_num:02d}{args.suffix}.dds"
        output_path = output_folder / filename
        
        image_to_dds(frame, output_path, args.width, args.height, args.fill, args.no_crop)
        output_files.append(filename)
    
    # Generate .seq file if requested
    if args.seq:
        seq_filename = f"{args.prefix}{args.start:02d}{args.suffix}.seq"
        seq_path = output_folder / seq_filename
        
        # Get the folder name for the path
        folder_name = output_folder.name
        
        with open(seq_path, 'w') as f:
            f.write("26\n")  # Animation speed
            for filename in output_files:
                # Remove .dds extension and write path with folder name
                name = filename[:-4]  # Remove .dds
                f.write(f"{args.seq_path}\\{folder_name}\\{name}\n")
    
    print(f"    Created {len(output_files)} DDS ({args.width}x{args.height})" + (" + SEQ" if args.seq else ""))


if __name__ == '__main__':
    main()
