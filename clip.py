import torch
from faster_whisper import WhisperModel

import subprocess
import json
import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
import string
import yt_dlp
from openai import OpenAI
import glob
from tqdm import tqdm
from tqdm import trange
import re
import pysrt
import glob
import pysubs2
import nltk
import ssl
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import argparse

# NLTK data downloads with error handling for remote environments
def ensure_nltk_data():
    """Ensure NLTK data is available, with fallbacks for remote environments"""
    
    # Set NLTK data path for remote environments
    import os
    nltk_data_paths = [
        '/workspace/nltk_data',  # Vast.ai custom path
        '/root/nltk_data',       # Docker containers
        os.path.expanduser('~/nltk_data')  # User home
    ]
    
    for path in nltk_data_paths:
        if os.path.exists(path):
            if path not in nltk.data.path:
                nltk.data.path.insert(0, path)
    
    try:
        # Try to download punkt tokenizer data
        print("📚 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # New punkt tokenizer
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️ NLTK download warning: {e}")
        # Try alternative download locations
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Try downloading to the first available path
            download_dir = next((p for p in nltk_data_paths if os.path.exists(os.path.dirname(p))), None)
            if download_dir:
                os.makedirs(download_dir, exist_ok=True)
                nltk.download('punkt', download_dir=download_dir, quiet=True)
                nltk.download('punkt_tab', download_dir=download_dir, quiet=True)
                nltk.download('stopwords', download_dir=download_dir, quiet=True)
                print("✅ NLTK data downloaded with SSL workaround")
            else:
                raise Exception("No suitable download directory found")
        except Exception as e2:
            print(f"⚠️ NLTK download failed: {e2}")
            print("ℹ️ Will use fallback sentence splitting if needed")

ensure_nltk_data()

# B-roll integration
try:
    from production_broll import ProductionBRollPipeline
    BROLL_AVAILABLE = True
    print("🎬 B-roll system available")
except ImportError:
    BROLL_AVAILABLE = False
    print("⚠️ B-roll system not available (production_broll.py not found)")



load_dotenv()

CLIPS_DIR = "clips"
CAPTIONS_DIR = "captions"
TRANSCRIPTS_DIR = "transcripts"
BROLL_DIR = "broll"  # New directory for B-roll videos
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(CAPTIONS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(BROLL_DIR, exist_ok=True)  # Create B-roll directory


client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)


elevenlabs = ElevenLabs(
  api_key= os.getenv("ELEVENLABS_API_KEY"),
)

# Initialize B-roll pipeline
broll_pipeline = None
if BROLL_AVAILABLE:
    try:
        from production_broll import create_production_pipeline
        broll_pipeline = create_production_pipeline()
        print("✅ B-roll pipeline initialized")
    except Exception as e:
        print(f"⚠️ B-roll pipeline initialization failed: {e}")
        BROLL_AVAILABLE = False
model = WhisperModel("small.en", device="cuda" if torch.cuda.is_available() else "cpu")

import json
import re

def safe_parse_gpt_response(response_text):
    # Regex to extract JSON array or object
    match = re.search(r'(\[.*\]|\{"clips":\s*\[.*?\]\})', response_text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in GPT response.")

    try:
        parsed = json.loads(match.group(1))
        return parsed["clips"] if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def detect_cuda_availability():
    """Check if CUDA/NVENC is available for FFmpeg"""
    try:
        result = subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1", 
            "-t", "1", "-c:v", "h264_nvenc", "-f", "null", "-"
        ], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def transcribe(video_path):
    segments, _ = model.transcribe(video_path, word_timestamps=True)

    transcript = ""
    segments_list = []
    for segment in segments:
        segment_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word
                } for word in segment.words
            ] if segment.words else []
        }
        segments_list.append(segment_dict)
        transcript += segment.text + " "

    transcript_path = os.path.join(TRANSCRIPTS_DIR, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript.strip())

    segments_path = os.path.join(TRANSCRIPTS_DIR, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments_list, f, indent=2, ensure_ascii=False)

    return transcript.strip(), segments_list



def create_grey_freeze_frame(input_clip, output_clip, duration):
    """
    Create a greyed-out freeze frame from the first frame of a video,
    scaled and cropped identically to overlay_captions().
    """
    # Get input video dimensions
    cap = cv2.VideoCapture(input_clip)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Set crop size to not exceed video dimensions (same as overlay_captions)
    pre_crop_height = min(1920, height)
    pre_crop_width = int(pre_crop_height * 9 / 16)
    pre_crop_width = min(pre_crop_width, width)

    speaker_x = detect_speaker_center(input_clip)
    crop_x = max(0, min(speaker_x - pre_crop_width // 2, width - pre_crop_width))
    crop_y = 0

    # FFmpeg video filter: extract first frame -> crop -> grey filter -> loop
    vf_filter = (
        f"select='eq(n,0)',loop=loop=-1:size=1:start=0,"
        f"crop={pre_crop_width}:{pre_crop_height}:{crop_x}:{crop_y},"
        f"format=gray,fps=30"
    )

    # Check CUDA availability and use appropriate encoder
    cuda_available = detect_cuda_availability()
    
    if cuda_available:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_clip,
            "-vf", vf_filter,
            "-t", str(duration),
            "-c:v", "h264_nvenc",
            "-preset", "p4",  # Higher quality NVENC preset
            "-cq", "18",  # Better quality setting
            "-pix_fmt", "yuv420p",
            output_clip
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_clip,
            "-vf", vf_filter,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "slow",  # Better quality preset
            "-crf", "18",  # Higher quality (lower CRF)
            "-pix_fmt", "yuv420p",
            output_clip
        ]
    
    subprocess.run(cmd, check=True)
    print(f"✅ Grey freeze frame created: {output_clip}")

def get_audio_duration(mp3_file):
    """
    Get duration of an MP3 audio file in seconds.
    """
    audio = AudioSegment.from_file(mp3_file)
    return audio.duration_seconds



def concat_videos(video_list, output_file):
    """
    Concatenate a list of videos into one output file.
    """
    with open("concat_list.txt", "w") as f:
        for vid in video_list:
            safe_path = os.path.abspath(vid).replace('\\', '/')
            f.write(f"file '{safe_path}'\n")

    # Check CUDA availability and use appropriate encoder
    cuda_available = detect_cuda_availability()
    
    if cuda_available:
        print("  🚀 Using CUDA hardware acceleration for video concatenation")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", "concat_list.txt",
            "-c:v", "h264_nvenc",
            "-preset", "p4",  # Higher quality NVENC preset
            "-cq", "18",  # Better quality setting
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_file
        ]
    else:
        print("  💻 Using CPU encoding for video concatenation (CUDA not available)")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", "concat_list.txt",
            "-c:v", "libx264",
            "-preset", "slow",  # Better quality preset
            "-crf", "18",  # Higher quality (lower CRF)
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_file
        ]
    
    subprocess.run(cmd, check=True)


def extract_clips(transcript, var ,max_clips=2):
    # Truncate transcript at the last sentence before 5000 chars
    prompt = f"""

       From the transcript below, extract {max_clips} short clips between 30-60 seconds that are likely to perform well on TikTok or Instagram Reels.
 
        Each clip must be completely self-contained:
        - Start at the **beginning of a complete sentence or clear idea**
        - Prefer clips that begin with a strong hook or can grab attention in the first 2 seconds

        For example, if a clip ends with “That’s the impact of cortisol,” 
        the clip must begin with the earliest point where cortisol or its symptoms are 
        first mentioned — not at the emotional peak.

        Example clips:
         GOOD CLIP:
        "Did you know there's a really fascinating experiment done on weight lifters?
          They lifted no weights for two weeks. 
          They just sat there and they visualized themselves lifting weights. They had a 13% increase in muscle mass. 
        People should realize how much potential they have in their brains."

         BAD CLIP:
        "...and that’s why it all comes down to dopamine. Because without it, you’re just not going to feel motivated to do anything."
        
         Prioritize clips that:
        - Contain a surprising fact, bold opinion, or viral insight
        - Feel emotionally powerful, inspiring, or funny
        - Could make someone stop scrolling within the first 2–3 seconds

        When providing the "transcript_text" for each clip:
        - Always copy the exact sentences from the transcript provided (verbatim).
        - Do not rewrite or paraphrase.

        Do not be afraid to use emojis in the hook.

        Return a JSON array like:
        [
        {{
            "start": "HH:MM:SS",
            "end": "HH:MM:SS",
            "hook": "1-sentence attention grabber",
            "caption": "3-line punchy caption",
            "transcript_text": "Full transcript text of the clip"
        }}
        ]

        Transcript:
        {transcript}
        """


    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        temperature=0.2,
        messages=[ 
            {"role": "system", "content": "   You are a smart short-form content editor with a talent for creating viral, Gen Z-friendly edutainment."},
            {"role": "user", "content": prompt}
        ],

    )
    print("GPT response:", response)
    with open(f"gpt_response{var}.txt", "w", encoding="utf-8") as f:
        f.write(response.choices[0].message.content)
    try:
        return safe_parse_gpt_response(response.choices[0].message.content)
    except Exception as e:
        print("❌ GPT response parsing failed:", e)
        print(response.choices[0].message.content)
        return []

def get_words_in_range(words, clip_start, clip_end):
    """Extract words from Whisper output within clip boundaries."""
    return [
        {
            "start": w["start"],
            "end": w["end"],
            "text": w["word"]
        }
        for w in words
        if clip_start <= w["start"] <= clip_end
    ]


def chunk_transcript(segments, chunk_duration=360, overlap=60):
    """Split transcript segments into chunks with overlap.

    Args:
        segments: List of Whisper segments.
        chunk_duration: Length of each chunk in seconds (default 6 minutes = 360s).
        overlap: Number of seconds of overlap between chunks.

    Returns:
        List of chunks (each is a list of segments).
    """
    chunks = []
    current_chunk = []
    chunk_start = 0
    current_time = 0

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        if current_time == 0:
            current_time = seg_start

        # If the segment fits in the current chunk
        if seg_end <= chunk_start + chunk_duration:
            current_chunk.append(seg)
        else:
            # Save current chunk
            chunks.append(current_chunk)
            # Move chunk start forward with overlap
            chunk_start += chunk_duration - overlap
            # Start a new chunk
            current_chunk = [seg]

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def extract_transcript_text(segments):
    raw_text = " ".join([seg["text"].strip() for seg in segments])
    
    # Try NLTK first, fallback to simple splitting
    try:
        sentences = sent_tokenize(raw_text)
    except LookupError:
        # NLTK data not available, use simple sentence splitting
        print("⚠️ NLTK punkt tokenizer not available, using fallback sentence splitting")
        # Simple sentence splitting on periods, exclamation marks, and question marks
        import re
        sentences = re.split(r'[.!?]+', raw_text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    return " ".join(sentences)


def hms_to_sec(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s  

    
def cut_clip(input_video, start, end, output_path):
    # Convert comma to dot for ffmpeg compatibility
    start = start.replace(',', '.')
    end = end.replace(',', '.')
    
    # Parse time strings that may contain milliseconds
    def parse_time_with_milliseconds(time_str):
        if '.' in time_str:
            # Has milliseconds: "HH:MM:SS.mmm"
            time_part, ms_part = time_str.split('.')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part.ljust(3, '0')[:3])  # Pad or truncate to 3 digits
            return h * 3600 + m * 60 + s + ms / 1000.0
        else:
            # No milliseconds: "HH:MM:SS"
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
    
    start_seconds = parse_time_with_milliseconds(start)
    end_seconds = parse_time_with_milliseconds(end)
    duration = str(end_seconds - start_seconds)
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", input_video,
        "-t", duration,
        "-c:v", "copy",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)


def generate_subs_from_whisper_segments(segments, output_file="captions.ass"):
    import pysubs2
    
    subs = pysubs2.SSAFile()
    
    # Orange style - for SUPER significant words only
    orange_style = pysubs2.SSAStyle()
    orange_style.fontname = "Montserrat Bold"
    orange_style.fontsize = 16
    orange_style.bold = True
    orange_style.italic = False
    orange_style.primarycolor = pysubs2.Color(255, 128, 0)  # Bright orange
    orange_style.outlinecolor = pysubs2.Color(0, 0, 0)
    orange_style.backcolor = pysubs2.Color(0, 0, 0, 0)
    orange_style.borderstyle = 1
    orange_style.outline = 2
    orange_style.shadow = 0.8
    orange_style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    orange_style.marginv = 50
    subs.styles["Orange"] = orange_style
    
    # White style - for everything else
    white_style = pysubs2.SSAStyle()
    white_style.fontname = "Montserrat Bold"
    white_style.fontsize = 16
    white_style.bold = True
    white_style.italic = False
    white_style.primarycolor = pysubs2.Color(255, 255, 255)  # White
    white_style.outlinecolor = pysubs2.Color(0, 0, 0)
    white_style.backcolor = pysubs2.Color(0, 0, 0, 0)
    white_style.borderstyle = 1
    white_style.outline = 2
    white_style.shadow = 0.8
    white_style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    white_style.marginv = 50
    subs.styles["White"] = white_style
    
    # Track spacing for orange highlights
    total_groups = len(segments) // 3 + (1 if len(segments) % 3 != 0 else 0)
    
    # Group words into groups of three
    for i in range(0, len(segments), 3):
        words_in_group = segments[i:i+3]
        group_index = i // 3
        
        # Start time from first word
        start_ms = int(words_in_group[0]['start'] * 1000)
        
        # End time from last word in group
        end_ms = int(words_in_group[-1]['end'] * 1000)
        
        # Combine all words in the group and make uppercase
        text = " ".join([remove_punctuation(word['text'].strip()) for word in words_in_group]).upper()
        
        # Determine style based on SUPER selective criteria
        style_name = analyze_semantic_importance_selective(words_in_group, group_index, total_groups)
        
        subs.append(pysubs2.SSAEvent(
            start=start_ms,
            end=end_ms,
            text=text,
            style=style_name
        ))
    
    subs.save(output_file)
    print(f"✅ SUPER selective semantic-based subtitles saved to {output_file}")


def remove_punctuation(text):
    """Remove all punctuation from the text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def wrap_text(text, max_words=3):  # Updated to 3 words
    """Wrap text into lines with at most max_words words, no punctuation."""
    text = remove_punctuation(text).upper()  # Make uppercase
    words = text.split()
    lines = []
    for i in range(0, len(words), max_words):
        lines.append(' '.join(words[i:i+max_words]))
    return '\\N'.join(lines)



def overlay_captions(video_file, ass_file, output_file):
    try:
        # Validate inputs
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        if not os.path.exists(ass_file):
            raise FileNotFoundError(f"ASS caption file not found: {ass_file}")
        
        if not output_file:
            raise ValueError("output_file cannot be empty")
        
        # Get input video dimensions
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_file}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

        # Convert ASS file path to FFmpeg-safe format
        ass_file_abs = os.path.abspath(ass_file)
        ass_file_ffmpeg = ass_file_abs.replace('\\', '/').replace(':', '\\\\:')

        # Check if video is already 9:16 format (portrait)
        is_already_portrait = height > width
        aspect_ratio = width / height if height > 0 else 1.0
        
        if is_already_portrait and abs(aspect_ratio - (9/16)) < 0.1:
            # Video is already 9:16, just apply captions and enhancement
            print(f"  📱 Video already 9:16 ({width}x{height}), applying captions only")
            vf_filter = (
                f"eq=contrast=1.2:saturation=1.5,"
                f"hue=s=1.1,"
                f"ass={ass_file_ffmpeg}"
            )
        else:
            # Video is 16:9, need to crop and scale - BUT DO FINAL SCALING ONLY
            print(f"  🔄 Converting {width}x{height} to 9:16 with captions")
            
            # Set crop size to not exceed video dimensions
            pre_crop_height = min(1920, height)
            pre_crop_width = int(pre_crop_height * 9 / 16)
            pre_crop_width = min(pre_crop_width, width)

            speaker_x = detect_speaker_center(video_file)
            crop_x = max(0, min(speaker_x - pre_crop_width // 2, width - pre_crop_width))
            crop_y = 0  # You can adjust this if you want vertical centering

            # QUALITY FIX: Crop but don't upscale yet - do final scaling only
            # This avoids double scaling that causes quality degradation
            print(f"  📏 Quality improvement: {width}x{height} → crop to {pre_crop_width}x{pre_crop_height} → scale to 1080x1920")
            vf_filter = (
                f"crop={pre_crop_width}:{pre_crop_height}:{crop_x}:{crop_y},"
                f"scale=1080:1920,"  # Scale to 1080p 9:16 for better quality/performance balance
                f"eq=contrast=1.2:saturation=1.5,"
                f"hue=s=1.1,"
                f"ass={ass_file_ffmpeg}"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_file,
            "-vf", vf_filter,
            "-c:v", "libx264",  # Use CPU encoding as fallback
            "-preset", "slow",  # Better quality preset
            "-crf", "18",  # Higher quality (lower CRF)
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            output_file
        ]

        # Try to detect if CUDA is available first
        cuda_available = detect_cuda_availability()
        
        if cuda_available:
            print("  🚀 Using CUDA hardware acceleration")
            cmd = [
                "ffmpeg", "-y",
                "-hwaccel", "cuda",
                "-i", video_file,
                "-vf", vf_filter,
                "-c:v", "h264_nvenc",
                "-preset", "p4",  # Higher quality NVENC preset (equivalent to slow)
                "-rc", "vbr",  # Variable bitrate for better quality
                "-cq", "18",   # Quality setting for NVENC (lower = better)
                "-b:v", "15M", # Higher bitrate for better quality
                "-maxrate", "20M",
                "-bufsize", "30M",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
                output_file
            ]
        else:
            print("  💻 Using CPU encoding (CUDA not available)")

        print("FFmpeg filter:", vf_filter)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            error_msg = f"FFmpeg failed (exit code {result.returncode}) for {output_file}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            
            # Clean up failed output
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except OSError:
                    pass
            
            raise RuntimeError(error_msg)
        
        # Verify output file was created and has content
        if not os.path.exists(output_file):
            raise RuntimeError(f"Output file was not created: {output_file}")
        
        output_size = os.path.getsize(output_file)
        if output_size < 10000:  # Less than 10KB
            raise RuntimeError(f"Output file too small ({output_size} bytes) - likely corrupted")
        
        print(f"✅ Successfully processed: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Caption overlay failed: {e}")
        # Clean up any partial output
        if 'output_file' in locals() and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError:
                pass
        raise  # Re-raise the exception
def is_super_significant_word(word):
    """
    Determine if a word is SUPER significant - the kind that makes people stop scrolling.
    These are high-impact, attention-grabbing words.
    """
    word_lower = word.lower().strip()  # Fixed: actually make it lowercase
    word_clean = word_lower.translate(str.maketrans('', '', string.punctuation))
    
    # Categories of super significant words
    
    # 1. Shock/Surprise words
    shock_words = {
        'shocking', 'incredible', 'unbelievable', 'amazing', 'insane', 'crazy', 
        'mind-blowing', 'stunning', 'jaw-dropping', 'explosive', 'revolutionary',
        'groundbreaking', 'breakthrough', 'miracle', 'impossible', 'forbidden', 
    }
    
    # 2. Extreme emotions/reactions
    extreme_emotions = {
        'terrifying', 'horrifying', 'devastating', 'catastrophic', 'tragic',
        'hilarious', 'outrageous', 'ridiculous', 'absurd', 'bizarre'
    }
    
    # 3. Power/Authority words
    power_words = {
        'secret', 'hidden', 'exposed', 'revealed', 'truth', 'lies', 'conspiracy',
        'scandal', 'exclusive', 'banned', 'censored', 'classified', 'leaked'
    }
    
    # 4. Urgency/Importance
    urgency_words = {
        'urgent', 'critical', 'essential', 'vital', 'crucial', 'emergency',
        'warning', 'danger', 'risk', 'threat', 'crisis', 'breaking'
    }
    
    # 5. Superlatives and extremes
    superlatives = {
        'biggest', 'largest', 'smallest', 'fastest', 'slowest', 'richest',
        'poorest', 'strongest', 'weakest', 'best', 'worst', 'first', 'last', 'fat',
        'ultimate', 'maximum', 'minimum', 'extreme', 'record', 'legendary'
    }
    
    # 6. Controversy/Conflict
    controversy_words = {
        'controversial', 'debate', 'argument', 'fight', 'battle', 'war',
        'conflict', 'opposite', 'against', 'versus', 'challenge', 'attack'
    }
    
    # 7. Money/Success/Failure
    money_success = {
        'money', 'rich', 'wealthy', 'millionaire', 'billionaire', 'poor',
        'broke', 'success', 'failure', 'bankrupt', 'profit', 'loss'
    }
    
    # 8. Science/Discovery words that are impactful
    science_impact = {
        'discovered', 'invented', 'created', 'destroyed', 'killed', 'saved',
        'cured', 'disease', 'cancer', 'brain', 'dna', 'genetic', 'experiment'
    }
    
    # 9. Health/Body/Wellness (relevant to your content)
    health_body = {
        'stress', 'cortisol', 'belly', 'weight', 'muscle', 'trauma', 'anxiety',
        'depression', 'hormones', 'periods', 'menstrual', 'sync', 'contagious',
        'visualization', 'manifest', 'neuroscience', 'brain', 'memory', 'learn'
    }
    
    # 10. Numbers that grab attention (when they're significant)
    if word_clean.isdigit():
        num = int(word_clean)
        # Large numbers, percentages, or significant small numbers
        if num >= 1000 or num >= 90 or num == 0:
            return True
    
    # 11. Percentage indicators
    if '%' in word or 'percent' in word_clean:
        return True
    
    # Combine all categories
    all_significant_words = (shock_words | extreme_emotions | power_words | 
                           urgency_words | superlatives | controversy_words | 
                           money_success | science_impact | health_body)
    
    return word_clean in all_significant_words

def analyze_semantic_importance_selective(words_group, group_index, total_groups):
    """
    Analyze a group of words for orange highlighting.
    Highlight if the group contains significant words.
    """
    # Check if any word in the group is super significant
    has_super_word = any(is_super_significant_word(word['text']) for word in words_group)
    
    if has_super_word:
        return "Orange"
    else:
        return "White"

def has_extraordinary_word(words_group):
    """
    Check for words that are SO significant they override the spacing rule.
    """
    extraordinary_words = {
        'secret', 'shocking', 'impossible', 'revolutionary', 'breakthrough',
        'miracle', 'forbidden', 'banned', 'exposed', 'truth', 'lies',
        'incredible', 'unbelievable', 'mind-blowing', 'terrifying',
        'millionaire', 'billionaire', 'killed', 'saved', 'cured', 'cancer'
    }
    
    for word in words_group:
        word_clean = word['text'].lower().strip().translate(str.maketrans('', '', string.punctuation))
        if word_clean in extraordinary_words:
            return True
    return False

def find_text_sequence_in_segments(target_text, all_segments):
    """
    Find the best matching sequence of words in transcript segments using fuzzy matching.
    Uses multiple strategies: exact matching, fuzzy matching, and key phrase detection.
    """
    import re
    from difflib import SequenceMatcher
    
    # Normalize the target text for matching
    target_words = []
    for word in target_text.split():
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word and len(clean_word) > 1:  # Skip single letters
            target_words.append(clean_word)
    
    if len(target_words) < 3:
        return None
    
    # Build list of all words with timestamps
    all_words = []
    for seg in all_segments:
        if 'words' in seg and seg['words']:
            for word_info in seg['words']:
                clean_word = re.sub(r'[^\w]', '', word_info['word'].lower())
                if clean_word and len(clean_word) > 1:
                    all_words.append({
                        'text': clean_word,
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'original': word_info['word']
                    })
    
    # Strategy 1: Try exact matching first
    best_match = None
    best_score = 0
    
    for i in range(len(all_words) - len(target_words) + 1):
        match_score = 0
        for j in range(len(target_words)):
            if i + j < len(all_words) and all_words[i + j]['text'] == target_words[j]:
                match_score += 1
            else:
                break
        
        score = match_score / len(target_words)
        if score > best_score:
            best_match = {
                'start_index': i,
                'end_index': i + match_score - 1,
                'score': score,
                'matched_words': match_score,
                'strategy': 'exact'
            }
            best_score = score
    
    # Strategy 2: If exact matching failed, try fuzzy matching with sliding window
    if best_score < 0.6:
        for i in range(len(all_words) - len(target_words) + 1):
            # Create a window of transcript words
            window_words = [all_words[i + j]['text'] for j in range(min(len(target_words), len(all_words) - i))]
            
            # Use SequenceMatcher for fuzzy comparison
            target_str = ' '.join(target_words)
            window_str = ' '.join(window_words)
            
            similarity = SequenceMatcher(None, target_str, window_str).ratio()
            
            if similarity > best_score:
                best_match = {
                    'start_index': i,
                    'end_index': i + len(window_words) - 1,
                    'score': similarity,
                    'matched_words': len(window_words),
                    'strategy': 'fuzzy'
                }
                best_score = similarity
    
    # Strategy 3: Key phrase detection - look for distinctive phrases
    if best_score < 0.5:
        # Extract key phrases (sequences of 3-5 words)
        for phrase_len in [5, 4, 3]:
            if len(target_words) >= phrase_len:
                for start_idx in range(len(target_words) - phrase_len + 1):
                    key_phrase = target_words[start_idx:start_idx + phrase_len]
                    
                    # Search for this key phrase in transcript
                    for i in range(len(all_words) - phrase_len + 1):
                        transcript_phrase = [all_words[i + j]['text'] for j in range(phrase_len)]
                        
                        # Check for exact phrase match
                        if key_phrase == transcript_phrase:
                            # Expand around the match
                            expand_start = max(0, i - start_idx)
                            expand_end = min(len(all_words) - 1, i + phrase_len + (len(target_words) - start_idx - phrase_len))
                            
                            score = phrase_len / len(target_words) * 1.2  # Boost for key phrase detection
                            
                            if score > best_score:
                                best_match = {
                                    'start_index': expand_start,
                                    'end_index': expand_end,
                                    'score': score,
                                    'matched_words': phrase_len,
                                    'strategy': 'key_phrase'
                                }
                                best_score = score
                            break
                    
                    if best_score > 0.7:  # Found good match, stop searching
                        break
    
    # Convert best match to time-based result

    if best_match and best_score >= 0.4:  # Lower threshold for better coverage
        start_idx = best_match['start_index']
        end_idx = best_match['end_index']
        
        # Extend match to natural sentence boundaries
        while end_idx < len(all_words) - 1:
            check_word = all_words[end_idx]['original']
            if check_word.endswith('.') or check_word.endswith('!') or check_word.endswith('?'):
                break
            end_idx += 1
            if end_idx - start_idx > len(target_words) * 1.5:  # Don't extend too far
                break
        
        return {
            'start_time': all_words[start_idx]['start'],
            'end_time': all_words[end_idx]['end'],
            'confidence': best_score,
            'matched_words': best_match['matched_words'],
            'strategy': best_match['strategy']
        }
    
    return None

def align_clip_times_to_segments(clip, all_segments):
    """
    Adjust clip's start/end times using improved fuzzy text matching.
    """
    # Try to find the text sequence in the segments
    match = find_text_sequence_in_segments(clip["transcript_text"], all_segments)
    
    if match and match['confidence'] >= 0.4:
        # Use the matched timing
        start_time = match['start_time']
        end_time = match['end_time']
        strategy = match.get('strategy', 'unknown')
        print(f"  ✅ Found {strategy} match with {match['confidence']:.1%} confidence ({match['matched_words']} words)")
    else:
        # Fallback to the original timestamps from the clip
        if "start" in clip and "end" in clip:
            start_time = hms_to_sec(clip["start"])
            end_time = hms_to_sec(clip["end"])
            print("  ⚠️ Using fallback timing (no precise match found)")
        else:
            # Last resort fallback
            start_time = all_segments[0]["start"] if all_segments else 0
            end_time = all_segments[-1]["end"] if all_segments else 60
            print("  ❌ Using default timing")
    
    clip["start_sec"] = start_time
    clip["end_sec"] = end_time
    clip["start"] = format_time(start_time)
    clip["end"] = format_time(end_time)
    return clip

    
def detect_speaker_center(video_path):
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []

    # Avoid division by zero and ensure at least 1 sample
    sample_every = max(1, frame_count // 30) if frame_count > 0 else 1

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in trange(total_frames, desc="Detecting faces"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = detector.process(rgb)
            detections = getattr(detection_result, "detections", None)
            if detections and len(detections) > 0:
                box = detections[0].location_data.relative_bounding_box
                x_center = (box.xmin + box.width / 2) * width
                results.append(x_center)

        frame_idx += 1

    cap.release()

    # Fallback to center if no faces detected
    if not results:
        return width // 2 if width else 360  # 360 is a safe fallback for 720px crop
    return int(np.median(results))




import os, glob, yt_dlp

def download_youtube_video(url, output_path="downloads"):
    os.makedirs(output_path, exist_ok=True)
    output_template = os.path.join(output_path, "%(title).40s.%(ext)s")

    # Try multiple strategies to bypass bot detection
    strategies = [
        # Strategy 1: Use cookies from browser
        {
            'cookiesfrombrowser': ('chrome',),
            'format': 'best[ext=mp4]/mp4/best',
            'outtmpl': output_template,
            'retries': 3,
            'fragment_retries': 3,
            'socket_timeout': 120,
            'nocheckcertificate': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        },
        # Strategy 2: Use cookies file if it exists
        {
            'cookiefile': 'cookies.txt' if os.path.exists('cookies.txt') else None,
            'format': 'best[ext=mp4]/mp4/best',
            'outtmpl': output_template,
            'retries': 3,
            'fragment_retries': 3,
            'socket_timeout': 120,
            'nocheckcertificate': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        },
        # Strategy 3: Basic download without cookies
        {
            'format': 'best[ext=mp4]/mp4/best',
            'outtmpl': output_template,
            'retries': 2,
            'fragment_retries': 2,
            'socket_timeout': 60,
            'nocheckcertificate': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        }
    ]

    print(f"📥 Downloading from YouTube: {url}")
    
    for i, ydl_opts in enumerate(strategies, 1):
        # Remove None values from options
        ydl_opts = {k: v for k, v in ydl_opts.items() if v is not None}
        
        print(f"  Trying strategy {i}/3...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                print("✅ Download complete.")
                break
            except Exception as e:
                print(f"  ❌ Strategy {i} failed: {str(e)[:100]}...")
                if i == len(strategies):
                    print(f"❌ All download strategies failed. Last error: {e}")
                    print("\n💡 To fix this issue:")
                    print("1. Install a browser extension to export YouTube cookies")
                    print("2. Save cookies as 'cookies.txt' in this directory")
                    print("3. Or use yt-dlp with --cookies-from-browser chrome")
                    return None

    # Find the downloaded file
    downloaded_files = sorted(
        glob.glob(os.path.join(output_path, "*.mp4")),
        key=os.path.getctime,
        reverse=True
    )
    
    if downloaded_files:
        print(f"✅ Downloaded: {os.path.basename(downloaded_files[0])}")
        return downloaded_files[0]
    else:
        print("❌ No downloaded files found")
        return None


def shift_ass_to_clip_start(ass_path, offset_seconds):
    subs = pysubs2.load(ass_path)
    for line in subs.events:
        line.start = max(0, line.start - int(offset_seconds * 1000))
        line.end = max(0, line.end - int(offset_seconds * 1000))

    subs.save(ass_path)


def create_video_with_broll_integration(original_video, broll_info, captions_file=None, output_path=None):
    """Create final video with B-roll segments overlaid on original video, maintaining perfect audio/caption sync"""
    
    try:
        # Validate inputs
        if not os.path.exists(original_video):
            raise FileNotFoundError(f"Original video not found: {original_video}")
        
        if output_path is None:
            raise ValueError("output_path cannot be None")
        
        # Get video duration and dimensions
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", original_video
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to probe video duration: {result.stderr}")
        
        if not result.stdout.strip():
            raise RuntimeError(f"No duration information found for video: {original_video}")
        
        try:
            total_duration = float(result.stdout.strip())
        except ValueError:
            raise RuntimeError(f"Invalid duration format: {result.stdout.strip()}")
        
        # Get video dimensions
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries", "stream=width,height",
            "-of", "csv=p=0", original_video
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to probe video dimensions: {result.stderr}")
        
        if not result.stdout.strip():
            raise RuntimeError(f"No dimension information found for video: {original_video}")
        
        try:
            dimensions = result.stdout.strip().split(',')
            video_width, video_height = int(dimensions[0]), int(dimensions[1])
        except (ValueError, IndexError):
            raise RuntimeError(f"Invalid dimension format: {result.stdout.strip()}")
        
        is_portrait = video_height > video_width
        
        print(f"    Video: {video_width}x{video_height} ({'9:16' if is_portrait else '16:9'})")
        
        if not broll_info:
            # No B-roll, just apply captions to original
            print(f"    📝 No B-roll, applying captions to original video...")
            if captions_file:
                return overlay_captions(original_video, captions_file, output_path)
            else:
                import shutil
                shutil.copy2(original_video, output_path)
                return True
        
        # Validate B-roll files exist
        missing_broll = []
        for broll in broll_info:
            if not os.path.exists(broll["path"]):
                missing_broll.append(broll["path"])
        
        if missing_broll:
            raise FileNotFoundError(f"B-roll files not found: {missing_broll}")
        
        # Filter and constrain B-roll segments
        # CONSTRAINT: B-roll every ~10 seconds, 1-2 seconds duration each
        # IMPORTANT: No B-roll in first 5 seconds to avoid sync issues
        valid_broll = []
        last_broll_time = -10.0  # This allows spacing calculation to work
        
        for broll in sorted(broll_info, key=lambda x: x["start_time"]):
            # CRITICAL FIX: Don't allow B-roll in first 5 seconds of clip
            if broll["start_time"] < 5.0:
                print(f"    ⚠️ Skipping B-roll at {broll['start_time']:.1f}s (too early, minimum 5s)")
                continue
                
            # Check 10-second spacing constraint
            if broll["start_time"] - last_broll_time >= 10.0:
                # Constrain duration to 1-2 seconds max
                max_duration = min(2.0, broll.get("duration", 2.0))
                constrained_end = min(
                    broll["start_time"] + max_duration,
                    broll.get("end_time", broll["start_time"] + max_duration),
                    total_duration
                )
                
                valid_broll.append({
                    "path": broll["path"],
                    "start_time": broll["start_time"],
                    "end_time": constrained_end,
                    "duration": constrained_end - broll["start_time"]
                })
                last_broll_time = broll["start_time"]
                print(f"    ✅ B-roll scheduled: {broll['start_time']:.1f}s-{constrained_end:.1f}s")
            else:
                print(f"    ⚠️ Skipping B-roll at {broll['start_time']:.1f}s (too close to previous at {last_broll_time:.1f}s)")
        
        print(f"    🎬 Using {len(valid_broll)} B-roll segments (filtered from {len(broll_info)} candidates)")
        
        if not valid_broll:
            # No valid B-roll after filtering
            print(f"    📝 No valid B-roll after filtering, applying captions to original...")
            if captions_file:
                return overlay_captions(original_video, captions_file, output_path)
            else:
                import shutil
                shutil.copy2(original_video, output_path)
                return True
        
        # Create composite video without captions first (using overlay approach)
        composite_video = output_path.replace('.mp4', '_composite_no_captions.mp4')
        
        # Build FFmpeg inputs - original video first, then all B-roll videos
        inputs = ["-i", original_video]
        for broll in valid_broll:
            inputs.extend(["-i", broll["path"]])
        
        # NEW APPROACH: Overlay B-roll on top of original video (maintains sync)
        print(f"🎭 B-roll overlay approach: {len(valid_broll)} overlays (maintains perfect sync)")
        
        # Debug: Print the overlay schedule
        for i, broll in enumerate(valid_broll):
            print(f"  Overlay {i+1}: B-roll at {broll['start_time']:.1f}s-{broll['end_time']:.1f}s ({broll['duration']:.1f}s)")
        
        # Build overlay filter chain
        filter_parts = []
        
        # Start with the original video as base
        current_label = "[0:v]"
        
        # Scale and prepare each B-roll overlay
        for i, broll in enumerate(valid_broll):
            broll_input = i + 1  # B-roll inputs start at 1
            broll_label = f"broll{i}"
            overlay_out = f"overlay{i}"
            
            # FIXED: Properly prepare B-roll video to play during the specified time window
            broll_duration = broll['end_time'] - broll['start_time']
            
            # Scale B-roll to match video dimensions and prepare timing
            scale_filter = f"scale={video_width}:{video_height}:force_original_aspect_ratio=increase,crop={video_width}:{video_height}"
            
            # CRITICAL FIX: Use setpts to delay the B-roll video to start at the correct time
            # The B-roll should start playing from time 0 of the B-roll file, but appear at start_time in the main timeline
            filter_parts.append(
                f"[{broll_input}:v]{scale_filter},fps=30,"
                f"trim=start=0:duration={broll_duration:.3f},"
                f"setpts=PTS+{broll['start_time']:.3f}/TB[{broll_label}]"
            )
            
            # Create overlay with enable condition to show only during the specified time window
            # This ensures the B-roll only appears during its designated time period
            overlay_filter = (
                f"{current_label}[{broll_label}]overlay=0:0:"
                f"enable='between(t,{broll['start_time']:.3f},{broll['end_time']:.3f})'[{overlay_out}]"
            )
            filter_parts.append(overlay_filter)
            current_label = f"[{overlay_out}]"
        
        # Final output from overlay chain is already in current_label
        composite_output_label = current_label.strip('[]')
        
        filter_complex = ";".join(filter_parts)
        print(f"    🔧 Filter chain: {filter_complex[:200]}..." if len(filter_complex) > 200 else f"    🔧 Filter chain: {filter_complex}")
        
        # Create composite video (without captions) - IMPROVED QUALITY SETTINGS
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{composite_output_label}]",
            "-map", "0:a",  # Keep original audio track (maintains perfect sync)
            "-c:a", "copy",  # Copy audio without re-encoding
            "-c:v", "libx264",
            "-preset", "slow",  # Better quality preset
            "-crf", "18",  # Higher quality (lower CRF)
            "-pix_fmt", "yuv420p",
            composite_video
        ]
        
        print("    🔧 Creating composite with B-roll overlays...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = f"Composite creation failed (exit code {result.returncode})"
            if result.stderr:
                error_msg += f". STDERR: {result.stderr}"
            if result.stdout:
                error_msg += f". STDOUT: {result.stdout}"
            
            print(f"    ❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Verify the composite file was created and has content
        if not os.path.exists(composite_video):
            raise RuntimeError(f"Composite video was not created: {composite_video}")
        
        composite_size = os.path.getsize(composite_video)
        if composite_size < 10000:  # Less than 10KB
            raise RuntimeError(f"Composite video too small ({composite_size} bytes) - likely corrupted")
        
        # Now apply captions to the composite video
        if captions_file:
            print("    📝 Applying captions to composite video...")
            captions_success = overlay_captions(composite_video, captions_file, output_path)
            
            if not captions_success:
                raise RuntimeError("Caption overlay failed")
        else:
            # No captions, just rename composite to final output
            import shutil
            try:
                shutil.move(composite_video, output_path)
            except Exception as e:
                raise RuntimeError(f"Failed to move composite video to output path: {e}")
            captions_success = True
        
        # Clean up temporary composite file
        if os.path.exists(composite_video) and composite_video != output_path:
            try:
                os.remove(composite_video)
            except OSError as e:
                print(f"    ⚠️ Warning: Could not remove temporary file {composite_video}: {e}")
        
        # Verify final output exists and has content
        if not os.path.exists(output_path):
            raise RuntimeError(f"Final output was not created: {output_path}")
        
        final_size = os.path.getsize(output_path)
        if final_size < 10000:  # Less than 10KB
            raise RuntimeError(f"Final output too small ({final_size} bytes) - likely corrupted")
        
        file_size = final_size / (1024 * 1024)  # MB
        print(f"    ✅ B-roll overlay video created: {output_path} ({file_size:.1f}MB)")
        print("    🎯 Perfect sync maintained: Original duration preserved, audio/captions aligned")
        return True
            
    except Exception as e:
        print(f"    ❌ Error in B-roll timeline creation: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up any temporary files
        if 'composite_video' in locals() and os.path.exists(composite_video) and composite_video != output_path:
            try:
                os.remove(composite_video)
            except OSError:
                pass
        
        raise  # Re-raise the exception instead of returning False


def main(video_path, output_dir=None):
    # Set up output directories
    global CLIPS_DIR, CAPTIONS_DIR, TRANSCRIPTS_DIR, BROLL_DIR
    
    if output_dir:
        CLIPS_DIR = os.path.join(output_dir, "clips")
        CAPTIONS_DIR = os.path.join(output_dir, "captions")
        TRANSCRIPTS_DIR = os.path.join(output_dir, "transcripts")
        BROLL_DIR = os.path.join(output_dir, "broll")
        
        # Create output directories
        os.makedirs(CLIPS_DIR, exist_ok=True)
        os.makedirs(CAPTIONS_DIR, exist_ok=True)
        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        os.makedirs(BROLL_DIR, exist_ok=True)
    
    #TODO COMMENT IN AND OUT WHILE TESTING
    print("[1] Transcribing...")
    transcript, segments = transcribe(video_path)
    
    # with open('transcripts/transcript.txt', encoding='utf-8') as f:
    #     transcript = f.read()
    # with open(os.path.join(TRANSCRIPTS_DIR, "segments.json"), "r", encoding="utf-8") as f:
    #     segments = json.load(f)

    segments = [s for s in segments if s["start"] >= 300]

    print("[2] Chunking transcript for first-pass extraction...")
    chunks = chunk_transcript(segments, chunk_duration=600)

    all_candidate_clips = []
    #TODO UNCOMMENT LATER
    for i, chunk in enumerate(tqdm(chunks, desc="Chunk Pass ")):
        chunk_text = extract_transcript_text(chunk)
        print(f"🔍 Extracting from chunk {i+1}")

        

        raw_clips = extract_clips(chunk_text, i)  # Can return dict or list
        chunk_start_sec = chunk[0]["start"] if chunk else 0

        # Handle both formats: dict with 'clips' or raw list
        if isinstance(raw_clips, dict) and "clips" in raw_clips:
            clips = raw_clips["clips"]
        elif isinstance(raw_clips, list):
            clips = raw_clips
        else:
            print(f"⚠️ Unexpected clip format in chunk {i+1}")
            clips = []

        for clip in clips:
            # Add chunk start time to each clip
            clip = align_clip_times_to_segments(clip, chunk)

            # clip_segment_text = [
            
            #     s["text"]
            #     for s in segments
            #     if s["start"] < clip_end_sec and s["end"] > clip_start_sec
            # ]
            # clip["transcript_text"] = " ".join(clip_segment_text)


        if clips:
            all_candidate_clips.extend(clips)
        else:
            print(f"⚠️ No valid clips extracted for chunk {i+1}")

    print(f"[3] Reranking {len(all_candidate_clips)} clips by viral potential...")

    MAX_CLIPS_FOR_RERANK = 30
    if len(all_candidate_clips) > MAX_CLIPS_FOR_RERANK:
        print(f"⚠️ Too many candidate clips ({len(all_candidate_clips)}), truncating to {MAX_CLIPS_FOR_RERANK} for reranking.")
        all_candidate_clips = all_candidate_clips[:MAX_CLIPS_FOR_RERANK]
    clips_json = json.dumps(all_candidate_clips, indent=2).replace("{", "{{").replace("}", "}}")
    rerank_prompt = (
            "You are an elite short-form video editor with a talent for creating **viral, Gen Z-friendly TikToks and Reels**.\n\n"
            "Your goal is to **RERANK the following clips** by their potential to go viral.\n\n"
            "✅ **What makes a clip viral** (rank these highest):\n"
            "- Starts with a **scroll-stopping hook** (bold claim, shocking fact, controversial opinion, or intriguing question).\n"
            "- Has an **emotional charge** (funny, inspiring, surprising, relatable, or infuriating).\n"
            "- Works even **out of context** (doesn’t require the whole video to make sense).\n"
            "- Delivers value **fast** (viewer understands why they should care within 2–3 seconds).\n\n"
            "❌ **What to deprioritize** (rank these lowest):\n"
            "- Long, slow setups.\n"
            "- Clips that require too much context or explanation.\n"
            "- Passive or generic statements.\n\n"
            "---\n\n"
            "🎯 **Important:**\n"
            "✅ **Reuse the provided \"start\", \"end\", \"hook\", \"caption\", and \"transcript_text\" verbatim. Do not rewrite or edit them.**\n"
            "Your only task is to **rerank the clips**.\n\n"
            "---\n\n"
            "🎯 **Return the top 20 mo st viral clips**\n\n"
            "Format:\n"
            "[\n"
            "{\n"
            "    \"start\": \"HH:MM:SS\", <-- reuse\n"
            "    \"end\": \"HH:MM:SS\", <-- reuse\n"
            "    \"hook\": \"...\",   <-- reuse\n"
            "    \"caption\": \"...\", <-- reuse\n"
            "    \"transcript_text\": \"...\" <-- reuse\n"
            "},\n"
            "...\n"
            "]\n\n"
            "Clips:\n" + clips_json
        )


    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a smart short-form content editor with a talent for creating viral, Gen Z-friendly edutainment."},
            {"role": "user", "content": rerank_prompt}
        ]
    )

    try:
        final_clips = safe_parse_gpt_response(response.choices[0].message.content)
        with open("final_clips.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(final_clips, indent=2, ensure_ascii=False))
    except Exception as e:
        print("❌ GPT Rerank response parsing failed:", e)
        print(response.choices[0].message.content)
        final_clips = []

    print(f"[4] Final selected clips: {len(final_clips)}")
    
  
    print(f"[4.5] Aligning {len(final_clips)} clips with transcript...")
    aligned_clips = []
    for i, clip in enumerate(final_clips):
        print(f"  Aligning clip {i+1}: '{clip['transcript_text'][:50]}...'")
        aligned_clip = align_clip_times_to_segments(clip, segments)
        aligned_clips.append(aligned_clip)
    
    # Save the aligned clips for reference
    with open("final_clips_aligned.json", "w", encoding="utf-8") as f:
        json.dump(aligned_clips, f, indent=2, ensure_ascii=False)
    print("✅ Aligned clips saved to final_clips_aligned.json")
    
    final_clips = aligned_clips  # Use aligned clips for processing

    final_clips = json.load(open("final_clips_aligned.json", "r", encoding="utf-8"))

    # B-roll configuration
    generate_broll = os.getenv("GENERATE_BROLL", "true").lower() == "true"
    print(f"🎬 B-roll generation: {'✅ Enabled' if generate_broll and BROLL_AVAILABLE else '❌ Disabled'}")

    for i, clip in enumerate(tqdm(final_clips, desc="Processing final clips")):
        start, end = clip["start"], clip["end"]
        out_file = os.path.join(CLIPS_DIR, f"clip_{i+1}.mp4")
        print(f"\n[5] Cutting clip {i+1}: {start} to {end}")
        print(f"    Hook: {clip['hook'][:60]}...")
        cut_clip(video_path, start, end, out_file)

        # Generate B-roll for this clip
        broll_files = []
        if generate_broll and BROLL_AVAILABLE:
            broll_files = generate_broll_for_clip(clip, segments, i+1)


        print("[6] Generating captions...")
        
        # Parse time strings that may contain milliseconds
        def parse_time_to_seconds(time_str):
            # Handle both comma and dot as decimal separators
            time_str = time_str.replace(',', '.')
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 3:
                    h, m, s_with_ms = parts
                    if '.' in s_with_ms:
                        s, ms = s_with_ms.split('.')
                        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, '0')[:3]) / 1000.0
                    else:
                        return int(h) * 3600 + int(m) * 60 + int(s_with_ms)
            return 0
        
        start_sec = parse_time_to_seconds(start)
        end_sec = parse_time_to_seconds(end)
        ass_file = os.path.join(CAPTIONS_DIR, f"clip_{i+1}.ass")
        clip_words = []
        for s in segments:
            if "words" in s:
                clip_words.extend(
                    get_words_in_range(s["words"], start_sec, end_sec)
                )
        
        # Adjust word timestamps to be relative to clip start
        clip_words_adjusted = []
        for word in clip_words:
            adjusted_word = word.copy()
            adjusted_word["start"] = word["start"] - start_sec
            adjusted_word["end"] = word["end"] - start_sec
            clip_words_adjusted.append(adjusted_word)
        
        generate_subs_from_whisper_segments(clip_words_adjusted, ass_file)
        
        # Step 1: Create 9:16 cropped video WITHOUT captions first
        cropped_video = os.path.join(CLIPS_DIR, f"clip_{i+1}_cropped_no_captions.mp4")
        print(f"[7a] Creating 9:16 cropped video for clip {i+1}")
        
        # Just crop to 9:16 without captions
        crop_cmd = [
            "ffmpeg", "-y", "-i", out_file,
            "-vf", "scale=2160:3840:force_original_aspect_ratio=increase,crop=2160:3840,setsar=1:1",
            "-c:a", "copy",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            cropped_video
        ]
        
        result = subprocess.run(crop_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to create cropped video: {result.stderr}")
            continue
        
        # Step 2: Integrate B-roll and apply captions to final composite
        final_out = os.path.join(CLIPS_DIR, f"clip_{i+1}_captioned.mp4")
        print(f"[7b] Creating timeline with B-roll and captions for clip {i+1}")
        
        try:
            video_created = create_video_with_broll_integration(
                cropped_video,     # Start with uncaptioned cropped video
                broll_files,       # B-roll segments with timing info
                ass_file,          # Apply captions to final composite
                final_out          # Final output
            )
            
            # Clean up intermediate cropped video
            if os.path.exists(cropped_video):
                os.remove(cropped_video)
            
            if video_created:
                if broll_files:
                    print(f"✅ Clip {i+1} completed with {len(broll_files)} B-roll segments: {final_out}")
                else:
                    print(f"✅ Clip {i+1} completed: {final_out}")
            else:
                raise RuntimeError(f"create_video_with_broll_integration returned False for clip {i+1}")
                
        except Exception as e:
            # Clean up intermediate cropped video on error
            if os.path.exists(cropped_video):
                os.remove(cropped_video)
            
            print(f"❌ Failed to create final video for clip {i+1}: {e}")
            import traceback
            traceback.print_exc()
            
            # Don't continue with other clips if this one failed
            raise RuntimeError(f"Clip {i+1} processing failed: {e}")
    
    print("\n🎉 All clips processed successfully!")
    print(f"📁 Video clips: {CLIPS_DIR}/")
    if BROLL_AVAILABLE and generate_broll:
        print(f"🎬 B-roll videos: {BROLL_DIR}/")
    print(f"📝 Captions: {CAPTIONS_DIR}/")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and process video clips')
    parser.add_argument('--video', type=str, help='Path to video file or YouTube URL')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for clips')
    
    args = parser.parse_args()
    
    # Use provided video or fallback to default
    if args.video:
        input_source = args.video
    else:
        input_source = 'https://www.youtube.com/watch?v=hCW2NHbWNwA&t=327s'
    
    if input_source.startswith("http://") or input_source.startswith("https://"):
        print(f"🌐 Processing YouTube URL: {input_source}")
        video_path = download_youtube_video(input_source)
        if video_path is None:
            print("❌ Failed to download video. Exiting.")
            exit(1)
        if not os.path.exists(video_path):
            print(f"❌ Downloaded video file not found: {video_path}")
            exit(1)
    else:
        print(f"📁 Processing local file: {input_source}")
        video_path = input_source
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            exit(1)

    print(f"✅ Using video file: {video_path}")
    try:
        main(video_path, args.output_dir)
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
