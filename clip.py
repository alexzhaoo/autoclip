import torch
import whisper
import subprocess
import json
import os
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
import pysubs2

load_dotenv()

CLIPS_DIR = "clips"
CAPTIONS_DIR = "captions"
TRANSCRIPTS_DIR = "transcripts"
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(CAPTIONS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

#TODO COMMENT IN AND OUT WHILE TESTING
model = whisper.load_model("medium.en", device="cuda" if torch.cuda.is_available() else "cpu")

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

def transcribe(video_path):
    result = model.transcribe(video_path, word_timestamps=False)


    transcript_path = os.path.join(TRANSCRIPTS_DIR, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])


    segments_path = os.path.join(TRANSCRIPTS_DIR, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(result.get("segments", []), f, indent=2, ensure_ascii=False)

    return result["text"], result.get("segments", [])


def extract_clips(transcript, var ,max_clips=8):
    # Truncate transcript at the last sentence before 5000 chars
    prompt = f"""

        From the transcript below, extract {max_clips} short clips (each under 60 seconds) that are likely to perform well on TikTok or Instagram Reels.

        Each clip must be completely self-contained:
        - Start at the **beginning of a complete sentence or clear idea**
        - Include enough **setup/context** in the first few seconds so the viewer understands what’s happening
        - End cleanly without cutting someone off mid-thought

        DO NOT start in the middle of a sentence, or pick a moment that’s confusing without prior information.
        Clips should start as close as possible to the emotionally impactful or insightful moment, while including just enough context to make sense.

        For example, if a clip ends with “That’s the impact of cortisol,” the clip must begin with the earliest point where cortisol or its symptoms are first mentioned — not at the emotional peak.

        Always ensure the clip is a **full mini-story**, not just the ending or reaction.

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

    

        Return a JSON array like:
        [
        {{
            "start": "HH:MM:SS",
            "end": "HH:MM:SS",
            "hook": "1-sentence attention grabber",
            "caption": "3-line punchy caption"
        }}
        ]

        Transcript:
        {transcript}
        """


    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
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
    return " ".join([seg["text"] for seg in segments])

    

    
def cut_clip(input_video, start, end, output_path):
    # Convert comma to dot for ffmpeg compatibility
    start = start.replace(',', '.')
    end = end.replace(',', '.')
    # Calculate duration
    from datetime import datetime
    fmt = "%H:%M:%S"
    tdelta = (
        datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
    ).total_seconds()
    duration = str(tdelta)
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

    style = pysubs2.SSAStyle()
    style.fontname = "Montserrat Regular"  # Or try Montserrat if available
    style.fontsize = 20
    style.bold = True
    style.italic = False

    style.primarycolor = pysubs2.Color(255, 255, 255)  # White text
    style.outlinecolor = pysubs2.Color(0, 0, 0)        # Black outline (optional)
    style.backcolor = pysubs2.Color(0, 0, 0, 0)        # Transparent

    style.borderstyle = 1  # Outline + shadow
    style.outline = 0.8    # Light outline
    style.shadow = 0.5     # Soft shadow

    style.alignment = pysubs2.Alignment.MIDDLE_CENTER  # Centered
    style.marginv = 100  # Push it slightly above the bottom edge



    subs.styles["Default"] = style

    for seg in segments:
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        words = remove_punctuation(seg['text'].strip()).split()
        chunks = [' '.join(words[i:i+2]) for i in range(0, len(words), 2)]


        duration = (end_ms - start_ms) // len(chunks) if chunks else 0
        for i, chunk in enumerate(chunks):
            chunk_start = start_ms + i * duration
            chunk_end = chunk_start + duration
            subs.append(pysubs2.SSAEvent(start=chunk_start, end=chunk_end, text=chunk))


    subs.save(output_file)


def remove_punctuation(text):
    """Remove all punctuation from the text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def wrap_text(text, max_words=2):
    """Wrap text into lines with at most max_words words, no punctuation."""
    text = remove_punctuation(text)
    words = text.split()
    lines = []
    for i in range(0, len(words), max_words):
        lines.append(' '.join(words[i:i+max_words]))
    return '\\N'.join(lines)



def overlay_captions(video_file, ass_file, output_file):
    # Get input video dimensions
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Set crop size to not exceed video dimensions
    pre_crop_height = min(1920, height)
    pre_crop_width = int(pre_crop_height * 9 / 16)
    pre_crop_width = min(pre_crop_width, width)

    speaker_x = detect_speaker_center(video_file)
    crop_x = max(0, min(speaker_x - pre_crop_width // 2, width - pre_crop_width))
    crop_y = 0  # You can adjust this if you want vertical centering

    # Convert ASS file path to FFmpeg-safe format
    ass_file_abs = os.path.abspath(ass_file)
    ass_file_ffmpeg = ass_file_abs.replace('\\', '/').replace(':', '\\\\:')

    # FFmpeg video filter: crop -> upscale -> overlay ASS
    vf_filter = (
        f"crop={pre_crop_width}:{pre_crop_height}:{crop_x}:{crop_y},"
        f"scale=2160:3840,"
        f"ass={ass_file_ffmpeg}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", video_file,
        "-vf", vf_filter,
        "-c:v", "h264_nvenc",
        "-preset", "slow",
        "-b:v", "12M",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_file
    ]

    print("FFmpeg filter:", vf_filter)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg error for {output_file}:\n{result.stderr}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return None




    
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



def download_youtube_video(url, output_path="downloads"):
    os.makedirs(output_path, exist_ok=True)
    output_template = os.path.join(output_path, "%(title).40s.%(ext)s")

    ydl_opts = {
        'cookiefile': 'cookies.txt',
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
    }

    print(f"📥 Downloading from YouTube: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("✅ Download complete.")

    downloaded_files = sorted(glob.glob(os.path.join(output_path, "*.mp4")), key=os.path.getctime, reverse=True)
    return downloaded_files[0] if downloaded_files else None
def hms_to_sec(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s

def shift_ass_to_clip_start(ass_path, offset_seconds):
    subs = pysubs2.load(ass_path)
    for line in subs.events:
        line.start -= int(offset_seconds * 1000)
        line.end -= int(offset_seconds * 1000)
    subs.save(ass_path)

def add_offset_to_hms(hms, offset_sec):
    """Add offset_sec to an HH:MM:SS string and return new HH:MM:SS string."""
    h, m, s = map(int, hms.split(":"))
    total = h * 3600 + m * 60 + s + int(offset_sec)
    # If you want to keep milliseconds, adjust accordingly
    new_h = total // 3600
    new_m = (total % 3600) // 60
    new_s = total % 60
    return f"{new_h:02d}:{new_m:02d}:{new_s:02d}"

def main(video_path):
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
            clip_start_hms = add_offset_to_hms(clip["start"], chunk_start_sec)
            clip_end_hms = add_offset_to_hms(clip["end"], chunk_start_sec)
            clip_start_sec = hms_to_sec(clip_start_hms)
            clip_end_sec = hms_to_sec(clip_end_hms)

            clip_segment_text = [
                s["text"]
                for s in segments
                if s["start"] < clip_end_sec and s["end"] > clip_start_sec
            ]
            clip["transcript_text"] = " ".join(clip_segment_text)
            clip["start_sec"] = clip_start_sec
            clip["end_sec"] = clip_end_sec
            clip["start"] = clip_start_hms
            clip["end"] = clip_end_hms

        if clips:
            all_candidate_clips.extend(clips)
        else:
            print(f"⚠️ No valid clips extracted for chunk {i+1}")

    print(f"[3] Reranking {len(all_candidate_clips)} clips by viral potential...")

    MAX_CLIPS_FOR_RERANK = 30
    if len(all_candidate_clips) > MAX_CLIPS_FOR_RERANK:
        print(f"⚠️ Too many candidate clips ({len(all_candidate_clips)}), truncating to {MAX_CLIPS_FOR_RERANK} for reranking.")
        all_candidate_clips = all_candidate_clips[:MAX_CLIPS_FOR_RERANK]
    rerank_prompt = f"""
    You are an expert short-form video editor. Given a list of clips, rerank them by their potential to go viral. 

    Weigh transcript text heavily in your ranking.

    Your ranking should reflect:
    - Entertainment value
    - Uniqueness or surprise
    - Emotional or intellectual impact
    - TikTok/Reels virality potential

    Only return the top 8.

    Format:
    [
    {{
        "start": "HH:MM:SS",
        "end": "HH:MM:SS",
        "hook": "...",
        "caption": "...",
        "transcript_text": "..."
    }},
    ...
    ]

    Clips:
    {json.dumps(all_candidate_clips, indent=2)}
    """


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

    final_clips = json.load(open("final_clips.json", "r", encoding="utf-8"))
    for i, clip in enumerate(tqdm(final_clips, desc="Processing final clips")):
        start, end = clip["start"], clip["end"]
        out_file = os.path.join(CLIPS_DIR, f"clip_{i+1}.mp4")
        print(f"[5] Cutting clip {i+1}: {start} to {end}")
        cut_clip(video_path, start, end, out_file)

        print("[6] Generating captions...")
        start_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], start.split(":")))
        end_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))
        clip_segments = [s for s in segments if s['start'] < end_sec and s['end'] > start_sec]
        ass_file = os.path.join(CAPTIONS_DIR, f"clip_{i+1}.ass")
        generate_subs_from_whisper_segments(clip_segments, ass_file)

        final_out = os.path.join(CLIPS_DIR, f"clip_{i+1}_captioned.mp4")
        print(f"[7] Overlaying captions on clip {i+1}")
        clip_start_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], clip["start"].split(":")))
        shift_ass_to_clip_start(ass_file, clip_start_sec)
        overlay_captions(out_file, ass_file, final_out)
        print(f"✅ Clip {i+1} processed and saved to {final_out}")
    print("All clips processed successfully!")



if __name__ == "__main__":
    input_source = 'https://www.youtube.com/watch?v=hCW2NHbWNwA&t=327s'
    if input_source.startswith("http://") or input_source.startswith("https://"):
        video_path = download_youtube_video(input_source)
    else:
        video_path = input_source

    main(video_path)
