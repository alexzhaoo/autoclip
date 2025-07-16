import torch
from faster_whisper import WhisperModel

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
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import subprocess





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


elevenlabs = ElevenLabs(
  api_key= os.getenv("ELEVENLABS_API_KEY"),
)
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

    cmd = [
        "ffmpeg", "-y",
        "-i", input_clip,
        "-vf", vf_filter,
        "-t", str(duration),
        "-c:v", "h264_nvenc",
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

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", "concat_list.txt",
        "-c:v", "h264_nvenc",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_file
    ]
    subprocess.run(cmd, check=True)


def extract_clips(transcript, var ,max_clips=8):
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
    sentences = sent_tokenize(raw_text)
    return " ".join(sentences)


def hms_to_sec(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s  

    
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

def add_offset_to_hms(hms, offset_sec):
    """Add offset_sec to an HH:MM:SS string and return new HH:MM:SS string."""
    h, m, s = map(int, hms.split(":"))
    total = h * 3600 + m * 60 + s + int(offset_sec)
    # If you want to keep milliseconds, adjust accordingly
    new_h = total // 3600
    new_m = (total % 3600) // 60
    new_s = total % 60
    return f"{new_h:02d}:{new_m:02d}:{new_s:02d}"

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
        f"eq=contrast=1.2:saturation=1.5,"
        f"hue=s=1.1,"
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
    
    print(f"✅ Successfully processed: {output_file}")
    return output_file
def is_super_significant_word(word):
    """
    Determine if a word is SUPER significant - the kind that makes people stop scrolling.
    These are high-impact, attention-grabbing words.
    """
    word_lower = word.upper().strip()
    word_clean = word_lower.translate(str.maketrans('', '', string.punctuation))
    
    # Categories of super significant words
    
    # 1. Shock/Surprise words
    shock_words = {
        'shocking', 'incredible', 'unbelievable', 'amazing', 'insane', 'crazy', 
        'mind-blowing', 'stunning', 'jaw-dropping', 'explosive', 'revolutionary',
        'groundbreaking', 'breakthrough', 'miracle', 'impossible', 'forbidden', 'women'
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
    
    # 9. Numbers that grab attention (when they're significant)
    if word_clean.isdigit():
        num = int(word_clean)
        # Large numbers, percentages, or significant small numbers
        if num >= 1000 or num >= 90 or num == 0:
            return True
    
    # 10. Percentage indicators
    if '%' in word or 'percent' in word_clean:
        return True
    
    # Combine all categories
    all_significant_words = (shock_words | extreme_emotions | power_words | 
                           urgency_words | superlatives | controversy_words | 
                           money_success | science_impact)
    
    return word_clean in all_significant_words

def analyze_semantic_importance_selective(words_group, group_index, total_groups):
    """
    Analyze a group of words with VERY selective orange highlighting.
    Only highlight if:
    1. Contains a SUPER significant word, AND
    2. We haven't highlighted recently (spacing rule)
    """
    # Check if any word in the group is super significant
    has_super_word = any(is_super_significant_word(word['text']) for word in words_group)
    
    if not has_super_word:
        return "White"
    
    # Spacing rule: Only allow orange roughly every 6 words (2 groups of 3)
    # But make exceptions for truly extraordinary words
    min_spacing = 2  # Minimum 2 groups (6 words) between orange highlights
    
    # Check if we should highlight based on spacing
    # This is a simplified version - in a real implementation, you'd track previous highlights
    if group_index % min_spacing == 0 or has_extraordinary_word(words_group):
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

def align_clip_times_to_segments(clip, chunk_segments):
    """
    Adjust clip's start/end times to absolute times using Whisper word-level timestamps.
    """
    start_text = clip["transcript_text"].split()[0].lower()
    end_text = clip["transcript_text"].split()[-1].lower()

    # Find first and last word timestamps
    start_time = None
    end_time = None

    for seg in chunk_segments:
        for word in seg["words"]:
            word_text = word["word"].lower().strip(string.punctuation)
            if start_time is None and word_text == start_text:
                start_time = word["start"]
            if word_text == end_text:
                end_time = word["end"]

    if start_time is None:
        start_time = chunk_segments[0]["start"]  # fallback
    if end_time is None:
        end_time = chunk_segments[-1]["end"]    # fallback

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

def shift_ass_to_clip_start(ass_path, offset_seconds):
    subs = pysubs2.load(ass_path)
    for line in subs.events:
        line.start = max(0, line.start - int(offset_seconds * 1000))
        line.end = max(0, line.end - int(offset_seconds * 1000))

    subs.save(ass_path)



def main(video_path):
    #TODO COMMENT IN AND OUT WHILE TESTING
    print("[1] Transcribing...")
    # transcript, segments = transcribe(video_path)
    
    with open('transcripts/transcript.txt', encoding='utf-8') as f:
        transcript = f.read()
    with open(os.path.join(TRANSCRIPTS_DIR, "segments.json"), "r", encoding="utf-8") as f:
        segments = json.load(f)

    segments = [s for s in segments if s["start"] >= 300]

    print("[2] Chunking transcript for first-pass extraction...")
    chunks = chunk_transcript(segments, chunk_duration=600)

    all_candidate_clips = []
    # for i, chunk in enumerate(tqdm(chunks, desc="Chunk Pass ")):
    #     chunk_text = extract_transcript_text(chunk)
    #     print(f"🔍 Extracting from chunk {i+1}")

        

    #     raw_clips = extract_clips(chunk_text, i)  # Can return dict or list
    #     chunk_start_sec = chunk[0]["start"] if chunk else 0

    #     # Handle both formats: dict with 'clips' or raw list
    #     if isinstance(raw_clips, dict) and "clips" in raw_clips:
    #         clips = raw_clips["clips"]
    #     elif isinstance(raw_clips, list):
    #         clips = raw_clips
    #     else:
    #         print(f"⚠️ Unexpected clip format in chunk {i+1}")
    #         clips = []

    #     for clip in clips:
    #         # Add chunk start time to each clip
    #         clip = align_clip_times_to_segments(clip, chunk)

    #         # clip_segment_text = [
            
    #         #     s["text"]
    #         #     for s in segments
    #         #     if s["start"] < clip_end_sec and s["end"] > clip_start_sec
    #         # ]
    #         # clip["transcript_text"] = " ".join(clip_segment_text)


    #     if clips:
    #         all_candidate_clips.extend(clips)
    #     else:
    #         print(f"⚠️ No valid clips extracted for chunk {i+1}")

    # print(f"[3] Reranking {len(all_candidate_clips)} clips by viral potential...")

    # MAX_CLIPS_FOR_RERANK = 120
    # if len(all_candidate_clips) > MAX_CLIPS_FOR_RERANK:
    #     print(f"⚠️ Too many candidate clips ({len(all_candidate_clips)}), truncating to {MAX_CLIPS_FOR_RERANK} for reranking.")
    #     all_candidate_clips = all_candidate_clips[:MAX_CLIPS_FOR_RERANK]
    # clips_json = json.dumps(all_candidate_clips, indent=2).replace("{", "{{").replace("}", "}}")
    # rerank_prompt = (
    #         "You are an elite short-form video editor with a talent for creating **viral, Gen Z-friendly TikToks and Reels**.\n\n"
    #         "Your goal is to **RERANK the following clips** by their potential to go viral.\n\n"
    #         "✅ **What makes a clip viral** (rank these highest):\n"
    #         "- Starts with a **scroll-stopping hook** (bold claim, shocking fact, controversial opinion, or intriguing question).\n"
    #         "- Has an **emotional charge** (funny, inspiring, surprising, relatable, or infuriating).\n"
    #         "- Works even **out of context** (doesn’t require the whole video to make sense).\n"
    #         "- Delivers value **fast** (viewer understands why they should care within 2–3 seconds).\n\n"
    #         "❌ **What to deprioritize** (rank these lowest):\n"
    #         "- Long, slow setups.\n"
    #         "- Clips that require too much context or explanation.\n"
    #         "- Passive or generic statements.\n\n"
    #         "---\n\n"
    #         "🎯 **Important:**\n"
    #         "✅ **Reuse the provided \"start\", \"end\", \"hook\", \"caption\", and \"transcript_text\" verbatim. Do not rewrite or edit them.**\n"
    #         "Your only task is to **rerank the clips**.\n\n"
    #         "---\n\n"
    #         "🎯 **Return the top 20 mo st viral clips**\n\n"
    #         "Format:\n"
    #         "[\n"
    #         "{\n"
    #         "    \"start\": \"HH:MM:SS\", <-- reuse\n"
    #         "    \"end\": \"HH:MM:SS\", <-- reuse\n"
    #         "    \"hook\": \"...\",   <-- reuse\n"
    #         "    \"caption\": \"...\", <-- reuse\n"
    #         "    \"transcript_text\": \"...\" <-- reuse\n"
    #         "},\n"
    #         "...\n"
    #         "]\n\n"
    #         "Clips:\n" + clips_json
    #     )


    # response = client.chat.completions.create(
    #     model="gpt-4.1-2025-04-14",
    #     messages=[
    #         {"role": "system", "content": "You are a smart short-form content editor with a talent for creating viral, Gen Z-friendly edutainment."},
    #         {"role": "user", "content": rerank_prompt}
    #     ]
    # )

    # try:
    #     final_clips = safe_parse_gpt_response(response.choices[0].message.content)
    #     with open("final_clips.json", "w", encoding="utf-8") as f:
    #         f.write(json.dumps(final_clips, indent=2, ensure_ascii=False))
    # except Exception as e:
    #     print("❌ GPT Rerank response parsing failed:", e)
    #     print(response.choices[0].message.content)
    #     final_clips = []

    # print(f"[4] Final selected clips: {len(final_clips)}")

    final_clips = json.load(open("final_clips.json", "r", encoding="utf-8"))
    for i, clip in enumerate(tqdm(final_clips, desc="Processing final clips")):
        start, end = clip["start"], clip["end"]
        out_file = os.path.join(CLIPS_DIR, f"clip_{i+1}.mp4")
        print(f"[5] Cutting clip {i+1}: {start} to {end}")
        cut_clip(video_path, start, end, out_file)

        """
        GENERATE NARRATION
        """
        # narration_file = f"narrations/clip_{i+1}.mp3"
        # narration_text = clip["hook"]
        # narration_audio = elevenlabs.text_to_speech.convert(
        #     voice_id="JBFqnCBsd6RMkjVDRZzb",
        #     output_format="mp3_44100_128",
        #     text=narration_text,
        #     model_id="eleven_multilingual_v2",
        # )
        # with open(narration_file, "wb") as f:
        #     for chunk in narration_audio:
        #         f.write(chunk)

        print("[6] Generating captions...")
        start_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], start.split(":")))
        end_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))
        ass_file = os.path.join(CAPTIONS_DIR, f"clip_{i+1}.ass")
        clip_words = []
        for s in segments:
            if "words" in s:
                clip_words.extend(
                    get_words_in_range(s["words"], start_sec, end_sec)
                )
        generate_subs_from_whisper_segments(clip_words, ass_file)
        final_out = os.path.join(CLIPS_DIR, f"clip_{i+1}_captioned.mp4")
        print(f"[7] Overlaying captions on clip {i+1}")
        clip_start_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], clip["start"].split(":")))
        shift_ass_to_clip_start(ass_file, clip_start_sec)
        overlay_captions(out_file, ass_file, final_out)

        # freeze_intro_file = f"clips/clip_{i+1}_intro.mp4"
        # freeze_duration = get_audio_duration(narration_file)
        # create_grey_freeze_frame(out_file, freeze_intro_file, freeze_duration)

        # # 📝 Hook captions
        # num_words = max(1, len(narration_text.split()))
        # chunk_duration = freeze_duration / num_words
        # hook_words = [
        #     {"start": i * chunk_duration,
        #     "end": (i + 1) * chunk_duration,
        #     "text": word}
        #     for i, word in enumerate(narration_text.split())
        # ]
        # hook_ass_file = os.path.join(CAPTIONS_DIR, f"clip_{i+1}_hook.ass")
        # generate_subs_from_whisper_segments(hook_words, hook_ass_file)

        # # Burn hook captions
        # intro_with_captions = os.path.join(CLIPS_DIR, f"clip_{i+1}_intro_with_captions.mp4")
        # overlay_captions(freeze_intro_file, hook_ass_file, intro_with_captions)

        # main_with_captions = os.path.join(CLIPS_DIR, f"clip_{i+1}_main_with_captions.mp4")
        # concat_videos([intro_with_captions, final_out], main_with_captions)

        print(f"✅ Clip {i+1} processed and saved to {final_out}")
    print("All clips processed successful*ly!")



if __name__ == "__main__":
    input_source = 'https://www.youtube.com/watch?v=hCW2NHbWNwA&t=327s'
    if input_source.startswith("http://") or input_source.startswith("https://"):
        video_path = download_youtube_video(input_source)
    else:
        video_path = input_source

    main(video_path)
