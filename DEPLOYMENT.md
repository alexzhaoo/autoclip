# AutoClipper Deployment Guide

Complete guide for deploying your AI-powered video clipping system with Wand 2.2 B-roll generation.

## 🎬 System Overview

Your AutoClipper system includes:
- **AI Video Analysis**: GPT-4 powered clip extraction and ranking
- **Advanced B-roll**: Wand 2.2 integration via VAST.ai
- **Smart Captions**: Context-aware subtitle generation with semantic highlighting
- **Format Optimization**: Automatic 9:16 conversion with face detection
- **Production Pipeline**: End-to-end automation from YouTube URL to viral clips

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/alexzhaoo/autoclip.git
cd autoclip
```

### 2. Environment Setup

#### Option A: Local Development
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

#### Option B: VAST.ai Cloud Deployment
```bash
# On VAST.ai instance
git clone https://github.com/alexzhaoo/autoclip.git
cd autoclip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
# Core API Keys (Required)
OPENAI_API_KEY=sk-proj-your-openai-key-here
ELEVENLABS_API_KEY=sk_your-elevenlabs-key-here

# B-roll Configuration (Optional but Recommended)
VAST_AI_ENDPOINT=https://your-vast-instance.com
VAST_AI_TOKEN=your-vast-ai-token

# Processing Configuration
MAX_CLIPS=10
GENERATE_BROLL=true
PREFER_QUALITY=true
```

### 4. Run the System

```bash
python clip.py
```

The system will:
1. Download the specified YouTube video
2. Transcribe using Whisper
3. Extract viral clips using GPT-4
4. Generate B-roll with Wand 2.2
5. Create final videos with captions

## 🔧 Detailed Configuration

### API Keys Setup

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com)
2. Create account and add billing
3. Generate API key
4. Add to `.env` file

#### ElevenLabs API Key
1. Visit [ElevenLabs](https://elevenlabs.io)
2. Sign up for account
3. Get API key from dashboard
4. Add to `.env` file

#### VAST.ai Setup (For High-Quality B-roll)
1. Create account at [VAST.ai](https://vast.ai)
2. Rent a GPU instance with:
   - **GPU**: RTX 4090 or A100 (recommended)
   - **RAM**: 32GB+ 
   - **Storage**: 100GB+
   - **PyTorch**: Pre-installed
3. Set up Wand 2.2 endpoint (see B-roll section below)

### B-roll Generation Setup

Your system supports multiple B-roll generation modes:

#### Mode 1: Wand 2.2 via VAST.ai (High Quality)
```env
VAST_AI_ENDPOINT=https://your-vast-instance.com:8000
VAST_AI_TOKEN=your-token
PREFER_QUALITY=true
```

**VAST.ai Instance Setup:**
```bash
# On your VAST.ai instance
git clone https://github.com/wandb/wand-2.2
cd wand-2.2
pip install -r requirements.txt

# Start the API server
python api_server.py --port 8000 --host 0.0.0.0
```

#### Mode 2: Enhanced Placeholders (Fallback)
```env
PREFER_QUALITY=false
```
Creates smart animated placeholders with:
- Content-aware colors
- Animated visual effects
- Prompt-based styling

### System Requirements

#### Minimum (Development)
- **CPU**: 4 cores
- **RAM**: 8GB
- **GPU**: Optional (CPU transcription)
- **Storage**: 10GB free space
- **Internet**: Stable connection for API calls

#### Recommended (Production)
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: RTX 3060+ (for fast transcription)
- **Storage**: 100GB+ SSD
- **Internet**: High-speed connection

#### VAST.ai Instance (B-roll Generation)
- **GPU**: RTX 4090 or A100
- **RAM**: 32GB+
- **Storage**: 100GB+
- **Bandwidth**: Unlimited

## 📦 Deployment Options

### Option 1: Local Development

**Pros:**
- Full control over environment
- No cloud costs for basic features
- Fast iteration and testing

**Cons:**
- Limited B-roll quality (placeholders only)
- Requires local GPU for optimal performance

**Setup:**
```bash
# Clone repository
git clone https://github.com/alexzhaoo/autoclip.git
cd autoclip

# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Configure .env with OpenAI + ElevenLabs keys
# Run
python clip.py
```

### Option 2: VAST.ai Full Cloud

**Pros:**
- High-quality Wand 2.2 B-roll generation
- Powerful GPU acceleration
- Scalable on-demand

**Cons:**
- Higher costs (~$1-3/hour for good GPUs)
- Requires cloud setup

**Setup:**
```bash
# Rent VAST.ai instance with RTX 4090
# Clone and setup autoclipper
git clone https://github.com/alexzhaoo/autoclip.git
cd autoclip
pip install -r requirements.txt

# Setup Wand 2.2 in parallel terminal
git clone https://github.com/wandb/wand-2.2
cd wand-2.2
pip install -r requirements.txt
python api_server.py --port 8000 --host 0.0.0.0 &

# Configure .env with all keys including VAST_AI_ENDPOINT
# Run autoclipper
cd ../autoclip
python clip.py
```

### Option 3: Hybrid Setup

**Pros:**
- Cost-effective
- Best of both worlds
- Flexible scaling

**Cons:**
- More complex setup
- Network latency considerations

**Setup:**
- Run main AutoClipper locally
- Use separate VAST.ai instance just for B-roll generation
- Configure `VAST_AI_ENDPOINT` to point to your B-roll server

## 🎯 Production Configuration

### High-Performance Settings

```env
# .env for production
OPENAI_API_KEY=your-key
ELEVENLABS_API_KEY=your-key
VAST_AI_ENDPOINT=your-endpoint
VAST_AI_TOKEN=your-token

# Optimized settings
MAX_CLIPS=20
GENERATE_BROLL=true
PREFER_QUALITY=true
```

### Cost Optimization

```env
# .env for cost-conscious setup
OPENAI_API_KEY=your-key
ELEVENLABS_API_KEY=your-key
# No VAST.ai keys = enhanced placeholders

# Conservative settings
MAX_CLIPS=10
GENERATE_BROLL=true
PREFER_QUALITY=false
```

## 📊 Usage Examples

### Basic Usage
```python
# Default behavior - processes video from URL in script
python clip.py
```

### Custom Video
```python
# Edit clip.py, change input_source variable
input_source = 'https://youtube.com/watch?v=YOUR_VIDEO_ID'
# or
input_source = '/path/to/your/video.mp4'
```

### Batch Processing
```python
# Process multiple videos
videos = [
    'https://youtube.com/watch?v=video1',
    'https://youtube.com/watch?v=video2',
    '/path/to/local/video.mp4'
]

for video in videos:
    input_source = video
    main(video)
```

## 🔍 Output Structure

After processing, you'll get:

```
clips/
├── clip_1.mp4              # Raw extracted clip
├── clip_1_captioned.mp4    # Final clip with captions + B-roll
├── clip_2.mp4
├── clip_2_captioned.mp4
└── ...

captions/
├── clip_1.ass             # Subtitle files
├── clip_2.ass
└── ...

broll/
├── clip_1_broll_1.mp4     # Generated B-roll segments
├── clip_1_broll_2.mp4
└── ...

transcripts/
├── transcript.txt          # Full transcript
└── segments.json          # Timestamped segments

# Analysis files
final_clips_aligned.json    # Selected clips with timing
gpt_response*.txt          # GPT analysis logs
```

## 🐛 Troubleshooting

### Common Issues

#### "OPENAI_API_KEY not set"
```bash
# Check .env file exists and has correct key
cat .env
# Verify no extra spaces or quotes around key
```

#### "CUDA not available"
```bash
# Install CUDA version of PyTorch
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### "FFmpeg not found"
```bash
# Windows: Download from https://ffmpeg.org/
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

#### "B-roll generation failed"
```bash
# Check VAST.ai endpoint is running
curl http://your-vast-endpoint:8000/health

# Verify API token is correct
echo $VAST_AI_TOKEN

# Check logs for specific error
```

#### "Whisper transcription slow"
```bash
# Use smaller model for faster processing
# Edit clip.py line: model = WhisperModel("tiny.en", ...)
# Or ensure CUDA is working for GPU acceleration
```

### Performance Optimization

#### Speed up transcription:
```python
# Use smaller Whisper model
model = WhisperModel("tiny.en", device="cuda")
```

#### Reduce API costs:
```python
# Limit clips processed
MAX_CLIPS_FOR_RERANK = 50  # Instead of 120
```

#### Optimize video quality:
```python
# Adjust FFmpeg settings in overlay_captions()
"-b:v", "8M",  # Lower bitrate for smaller files
```

## 💰 Cost Estimates

### Per 1-hour video processing:

#### APIs:
- **OpenAI GPT-4**: ~$2-5 (depends on transcript length)
- **ElevenLabs**: ~$0.10 (if using voice generation)
- **Whisper**: Free (local processing)

#### Infrastructure:
- **Local**: $0 (electricity only)
- **VAST.ai RTX 4090**: ~$2-4/hour
- **VAST.ai A100**: ~$4-8/hour

#### Total per video:
- **Basic (local)**: $2-5
- **High-quality (VAST.ai)**: $4-12

## 🔐 Security Considerations

1. **API Keys**: Never commit `.env` files to git
2. **VAST.ai**: Use firewall rules to restrict access
3. **Output**: Review generated content before publishing
4. **Billing**: Set OpenAI usage limits to prevent overcharges

## 📈 Scaling for Production

### Batch Processing Setup
```python
# Create a queue-based system
import queue
import threading

video_queue = queue.Queue()
# Add videos to queue
# Process with multiple threads
```

### Monitoring
```python
# Add logging
import logging
logging.basicConfig(level=logging.INFO)

# Track costs
api_costs = {"openai": 0, "elevenlabs": 0}
```

### Database Integration
```python
# Store results in database
import sqlite3
# Track processed videos, avoid duplicates
```

## 🆘 Support

If you encounter issues:

1. **Check logs**: Look for error messages in console output
2. **Verify setup**: Ensure all API keys are correct
3. **Test components**: Run individual parts (transcription, B-roll, etc.)
4. **Resource limits**: Check disk space, memory usage
5. **Network issues**: Verify internet connectivity for API calls

## 🚀 Next Steps

Once deployed successfully:

1. **Optimize prompts**: Tune GPT prompts for your content style
2. **Custom B-roll**: Train your own models for specific content types
3. **Automation**: Set up scheduled processing for regular content
4. **Analytics**: Track which clips perform best
5. **Scaling**: Move to dedicated servers for higher volume

---

**Happy clipping! 🎬✨**

Your AutoClipper system is now ready to transform long-form content into viral short clips with AI-powered B-roll generation.
