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

load_dotenv()

CLIPS_DIR = "clips"
SRTS_DIR = "srts"
TRANSCRIPTS_DIR = "transcripts"
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(SRTS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

#TODO COMMENT IN AND OUT WHILE TESTING
model = whisper.load_model("small")

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


def extract_clips(transcript, max_clips=5):
    # Truncate transcript at the last sentence before 5000 chars
    prompt = f"""


    From the transcript below, extract {max_clips} compelling moments under 60 seconds each that would perform well on platforms like TikTok or Instagram Reels.

    Prioritize:
    - Interesting facts, unique perspectives, or surprising insights
    - Entertaining or emotionally resonant delivery
    - Content that feels like a “knowledge bomb” or “hot take”
    - Statements that make viewers *stop scrolling* and say “wait, what?”

    Format the output as JSON with:
    - start (HH:MM:SS)
    - end (HH:MM:SS)
    - hook (a 1-line opening to grab attention)
    - caption (3-line, punchy text — keep it tight and readable)

    Transcript:
    {transcript}
    """


    response = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=[
            {"role": "system", "content": "   You are a smart short-form content editor with a talent for creating viral, Gen Z-friendly edutainment."},
            {"role": "user", "content": prompt}
        ],

    )
    print("GPT response:", response)
    with open("gpt_response.txt", "w", encoding="utf-8") as f:
        f.write(response.choices[0].message.content)
    try:
        return safe_parse_gpt_response(response.choices[0].message.content)
    except Exception as e:
        print("❌ GPT response parsing failed:", e)
        print(response.choices[0].message.content)
        return []


def chunk_transcript(segments, chunk_duration=600):
    """Split transcript segments into chunks of `chunk_duration` seconds."""
    chunks = []
    current_chunk = []
    start_time = 0

    for seg in segments:
        if seg["start"] < start_time + chunk_duration:
            current_chunk.append(seg)
        else:
            chunks.append(current_chunk)
            current_chunk = [seg]
            start_time = seg["start"]

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



def generate_srt_from_caption(caption, output_file="captions.srt"):
    lines = caption.strip().split('\n')
    with open(output_file, "w") as f:
        for i, line in enumerate(lines):
            start = format_time(i * 3)
            end = format_time((i + 1) * 3)
            f.write(f"{i+1}\n{start} --> {end}\n{line.strip()}\n\n")

def remove_punctuation(text):
    """Remove all punctuation from the text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def wrap_text(text, max_words=6):
    """Wrap text into lines with at most max_words words, no punctuation."""
    text = remove_punctuation(text)
    words = text.split()
    lines = []
    for i in range(0, len(words), max_words):
        lines.append(' '.join(words[i:i+max_words]))
    return '\n'.join(lines)

def generate_srt_from_whisper_segments(segments, output_file="captions.srt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            start = format_time(seg['start'])
            end = format_time(seg['end'])
            # Remove punctuation and wrap text for better subtitle readability
            text = wrap_text(seg['text'].strip(), max_words=6)
            f.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")

def overlay_captions(video_file, srt_file, output_file):
    speaker_x = detect_speaker_center(video_file)
    crop_x = max(0, speaker_x - 540)  
    srt_file_abs = os.path.abspath(srt_file).replace('\\', '/').replace(':', '\\:')  # Escape colons for ffmpeg
    vf_filter = (
        f"scale=-1:1920,"
        f"crop=1080:1920:{crop_x}:0,"
        f"subtitles=\"{srt_file_abs}\""
    )

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", video_file,
        "-vf", vf_filter,
        "-c:v", "h264_nvenc",
        "-b:v", "4M",
        "-c:a", "copy",
        output_file
    ]
    print(vf_filter)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error for {output_file}:\n{result.stderr}")
        if os.path.exists(output_file):
            os.remove(output_file)  # clean up bad file
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
    # transcript, segments = transcribe(video_path)
    
    with open('transcripts/transcript.txt', encoding='utf-8') as f:
        transcript = f.read()
    with open(os.path.join(TRANSCRIPTS_DIR, "segments.json"), "r", encoding="utf-8") as f:
        segments = json.load(f)

    print("[2] Chunking transcript for first-pass extraction...")
    chunks = chunk_transcript(segments, chunk_duration=600)

    all_candidate_clips = []
    for i, chunk in enumerate(tqdm(chunks, desc="Chunk Pass ")):
        chunk_text = extract_transcript_text(chunk)
        print(f"🔍 Extracting from chunk {i+1}")

        raw_clips = extract_clips(chunk_text)  # Can return dict or list
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
    You are an expert short-form video editor. Given a list of clips, rerank them by their potential to go viral. Consider the *hook*, *caption*, and the full *transcript text* of the moment.

    Your ranking should reflect:
    - Entertainment value
    - Uniqueness or surprise
    - Emotional or intellectual impact
    - TikTok/Reels virality potential

    Only return the top 6.

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
        model="o4-mini-2025-04-16",
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

    for i, clip in enumerate(tqdm(final_clips, desc="Processing final clips")):
        start, end = clip["start"], clip["end"]
        out_file = os.path.join(CLIPS_DIR, f"clip_{i+1}.mp4")
        print(f"[5] Cutting clip {i+1}: {start} to {end}")
        cut_clip(video_path, start, end, out_file)

        print("[6] Generating captions...")
        start_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], start.split(":")))
        end_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))
        clip_segments = [s for s in segments if s['start'] < end_sec and s['end'] > start_sec]
        srt_file = os.path.join(SRTS_DIR, f"clip_{i+1}.srt")
        generate_srt_from_whisper_segments(clip_segments, srt_file)

        final_out = os.path.join(CLIPS_DIR, f"clip_{i+1}_captioned.mp4")
        print(f"[7] Overlaying captions on clip {i+1}")
        overlay_captions(out_file, srt_file, final_out)
        print(f"✅ Clip {i+1} processed and saved to {final_out}")
    print("All clips processed successfully!")



if __name__ == "__main__":
    input_source = 'https://www.youtube.com/watch?v=d7sUWwHugg8&t=321s'
    if input_source.startswith("http://") or input_source.startswith("https://"):
        video_path = download_youtube_video(input_source)
    else:
        video_path = input_source
    
    main(video_path)
