import os
import random
import subprocess
import glob
import argparse

def add_music_to_clips(clips_dir="clips", music_dir="music", output_dir="final_clips", target_lufs=-25.0):
    """
    Overlays random background music from music_dir onto clips in clips_dir.
    Uses loudnorm to normalize music to a specific LUFS target.
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")

    # Get list of captioned clips
    # Matches clip_X_captioned.mp4 pattern
    clip_files = glob.glob(os.path.join(clips_dir, "*_captioned.mp4"))
    
    if not clip_files:
        print(f"❌ No captioned clips found in '{clips_dir}'")
        print(f"   Looking for files matching: {os.path.join(clips_dir, '*_captioned.mp4')}")
        return

    # Get list of music files
    music_files = glob.glob(os.path.join(music_dir, "*.mp3"))
    if not music_files:
        print(f"❌ No .mp3 files found in '{music_dir}'")
        print("   Please add some MP3 files to the music folder and try again.")
        return

    print(f"🎬 Found {len(clip_files)} clips and {len(music_files)} music tracks.")
    print(f"🔊 Target Loudness: {target_lufs} LUFS (Dynamic Normalization)")

    for clip_path in clip_files:
        filename = os.path.basename(clip_path)
        # Create new filename: clip_1_captioned.mp4 -> clip_1_final.mp4
        output_filename = filename.replace("_captioned.mp4", "_final.mp4")
        output_path = os.path.join(output_dir, output_filename)

        # Pick random music track
        music_path = random.choice(music_files)
        music_name = os.path.basename(music_path)

        print(f"\n🎵 Processing: {filename}")
        print(f"   + Music: {music_name}")

        # FFmpeg command explanation:
        # -i clip_path: Input video (stream 0)
        # -stream_loop -1: Loop the music indefinitely
        # -i music_path: Input music (stream 1)
        # -filter_complex:
        #   [1:a]loudnorm=I={target_lufs}:TP=-1.5:LRA=11[bgm] -> Normalize music to target LUFS
        #   [0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[a] -> Mix original audio [0:a] and normalized music [bgm]
        # -map 0:v: Use video from first input
        # -map "[a]": Use our mixed audio
        # -c:v copy: Copy video stream directly
        # -c:a aac: Encode audio to AAC
        
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-stream_loop", "-1", 
            "-i", music_path,
            "-filter_complex",
            f"[1:a]loudnorm=I={target_lufs}:TP=-1.5:LRA=11[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]

        try:
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Saved to: {output_path}")
            else:
                print(f"❌ FFmpeg error for {filename}:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("\n✨ All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add background music to video clips")
    parser.add_argument("--target_lufs", type=float, default=-25.0, help="Target loudness in LUFS (default -25.0 for background music)")
    parser.add_argument("--clips_dir", type=str, default="clips/clippyths", help="Directory containing source clips")
    parser.add_argument("--music_dir", type=str, default="music", help="Directory containing music mp3s")
    parser.add_argument("--output_dir", type=str, default="final_clips", help="Directory for output files")
    
    args = parser.parse_args()

    add_music_to_clips(
        clips_dir=args.clips_dir,
        music_dir=args.music_dir,
        output_dir=args.output_dir,
        target_lufs=args.target_lufs
    )
