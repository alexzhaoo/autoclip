#!/usr/bin/env python3
"""
Add watermark to completed video clips.
Reads videos from clips/clips/ and applies watermark from post_processing/
Outputs watermarked videos to clips/clips_watermarked/
"""

import os
import sys
import subprocess
from pathlib import Path


def find_watermark(watermark_dir="post_processing"):
    """Find the watermark image in the post_processing directory"""
    watermark_path = Path(watermark_dir)
    
    if not watermark_path.exists():
        raise FileNotFoundError(f"Watermark directory not found: {watermark_dir}")
    
    # Look for common image formats
    for ext in ['.png', '.jpg', '.jpeg', '.svg']:
        watermark_files = list(watermark_path.glob(f"*{ext}"))
        if watermark_files:
            return str(watermark_files[0])
    
    raise FileNotFoundError(f"No watermark image found in {watermark_dir}")


def detect_cuda_availability():
    """Check if CUDA/NVENC is available"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def add_watermark_to_video(input_video, watermark_path, output_video, watermark_height=200, bottom_padding=50, use_gpu=True):
    """Add watermark to video at middle bottom position"""
    
    print(f"🎬 Processing: {os.path.basename(input_video)}")
    
    # Check GPU availability
    cuda_available = use_gpu and detect_cuda_availability()
    
    # FIX IS HERE: Added ':shortest=1' to the overlay options.
    # This tells FFmpeg to stop encoding when the video (input 0) ends, 
    # ignoring the infinite loop of the watermark (input 1).
    # Also using scale=-2 to ensure even width (important for some codecs)
    filter_graph = (
        f"[1:v]scale=-2:{watermark_height}[wm];"
        f"[0:v][wm]overlay=(W-w)/2:H-h-{bottom_padding}:shortest=1[outv]"
    )
    
    if cuda_available:
        print(f"    🚀 Using GPU acceleration (NVENC)")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-loop", "1", "-i", watermark_path,
            "-filter_complex", filter_graph,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "18",
            "-b:v", "10M",
            "-maxrate", "15M",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_video
        ]
    else:
        print(f"    💻 Using CPU encoding (GPU not available)")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-loop", "1", "-i", watermark_path,
            "-filter_complex", filter_graph,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_video
        ]
    
    print(f"    🛠️  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ❌ Failed: {result.stderr}")
        return False
    
    if not os.path.exists(output_video):
        print(f"    ❌ Output file not created")
        return False
    
    output_size = os.path.getsize(output_video) / (1024 * 1024)
    print(f"    ✅ Created: {os.path.basename(output_video)} ({output_size:.1f}MB)")
    return True


def process_all_videos(input_dir="clips/clips", watermark_dir="post_processing", output_dir="clips/clips_watermarked", watermark_height=200, bottom_padding=50, use_gpu=True):
    """Process all videos in the input directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find watermark
    try:
        watermark_path = find_watermark(watermark_dir)
        print(f"🖼️  Using watermark: {watermark_path}")
        print(f"📏 Settings: Height={watermark_height}px, Padding={bottom_padding}px\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Find all video files
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    video_files = []
    for ext in ['.mp4', '.mov', '.avi']:
        video_files.extend(input_path.glob(f"*{ext}"))
    
    if not video_files:
        print(f"⚠️  No video files found in {input_dir}")
        return
    
    print(f"📹 Found {len(video_files)} videos to process\n")
    
    # Process each video
    success_count = 0
    for video_file in video_files:
        output_file = os.path.join(output_dir, video_file.name)
        
        if add_watermark_to_video(str(video_file), watermark_path, output_file, watermark_height, bottom_padding, use_gpu):
            success_count += 1
    
    print(f"\n✅ Successfully watermarked {success_count}/{len(video_files)} videos")
    print(f"📂 Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add watermark to video clips")
    parser.add_argument("--input_dir", default="final_clips", help="Input directory with videos")
    parser.add_argument("--watermark_dir", default="post_processing", help="Directory containing watermark image")
    parser.add_argument("--output_dir", default="final_clips", help="Output directory for watermarked videos")
    parser.add_argument("--watermark_height", type=int, default=400, help="Height of watermark in pixels")
    parser.add_argument("--bottom_padding", type=int, default=850, help="Padding from bottom in pixels")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration and use CPU only")
    
    args = parser.parse_args()
    
    process_all_videos(
        args.input_dir, 
        args.watermark_dir, 
        args.output_dir, 
        args.watermark_height, 
        args.bottom_padding, 
        use_gpu=not args.no_gpu
    )