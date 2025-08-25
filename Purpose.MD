# AutoClipper: AI-Powered Viral Video Generation Pipeline

## Overview

AutoClipper is an automated system that transforms long-form video content (podcasts, lectures, interviews) into viral-ready short-form clips optimized for TikTok, Instagram Reels, and YouTube Shorts. The system uses AI to identify the most engaging moments, generate B-roll footage, and create polished vertical videos with dynamic captions.

## Core Functionality

### 🎯 What It Does
- **Input**: Long-form video (YouTube URL or local file)
- **Output**: 20 viral-optimized short clips (30-60 seconds each) in 9:16 format
- **Processing**: Fully automated with AI-driven content analysis and video generation

### 🧠 AI-Powered Content Analysis
The system uses OpenAI's GPT-4 to:
- Analyze video transcripts for viral potential
- Identify "scroll-stopping hooks" and engaging moments
- Extract clips that work standalone without context
- Rank content by Gen Z appeal and social media virality

## System Architecture

### Primary Components

#### 1. **clip.py** - Main Processing Pipeline
The core engine that orchestrates the entire workflow:

**Phase 1: Content Ingestion**
- Downloads YouTube videos using `yt-dlp`
- Supports local video file processing
- Handles various video formats and qualities

**Phase 2: Transcription & Analysis**
- Uses Faster Whisper for accurate speech-to-text with word-level timestamps
- Chunks long transcripts for efficient AI processing
- Employs GPT-4 for intelligent moment identification

**Phase 3: Clip Extraction**
- Identifies 8 candidate clips per transcript chunk
- Performs secondary AI ranking for viral potential
- Aligns AI-selected text with precise audio timestamps
- Extracts top 20 clips based on engagement metrics

**Phase 4: Visual Enhancement**
- Crops videos to 9:16 aspect ratio (portrait mode)
- Detects speaker position using MediaPipe face detection
- Applies visual enhancements (contrast, saturation)
- Generates AI B-roll footage using Wan2.2 model

**Phase 5: Caption Generation**
- Creates word-level captions with precise timing
- Implements semantic highlighting (orange for key words)
- Generates .ass subtitle files with custom styling
- Applies captions as burned-in overlays

#### 2. **run_on_vast.py** - Cloud Automation Script
Automates the entire pipeline on high-performance cloud GPUs:

**Instance Management**
- Launches RTX 5090 GPU instances on Vast.ai
- Configures PyTorch environment automatically
- Handles SSH connectivity and file transfers

**Dependency Installation**
- Downloads 5B parameter Wan2.2 AI video model
- Installs FFmpeg, Python libraries, and dependencies
- Sets up environment variables and paths

**Execution & Cleanup**
- Runs the full pipeline remotely
- Downloads processed results to local machine
- Automatically destroys instances to minimize costs

#### 3. **production_broll.py** - AI Video Generation
Generates contextual B-roll footage to enhance clips:

**Content Analysis**
- Analyzes transcript segments for visual opportunities
- Identifies moments suitable for B-roll insertion
- Generates prompts for AI video creation

**Video Generation**
- Uses Wan2.2 Text-to-Video model for B-roll creation
- Creates 2-3 second video segments
- Maintains quality and relevance to content

**Timeline Integration**
- Inserts B-roll at optimal intervals (~7 seconds apart)
- Maintains audio continuity while adding visual variety
- Applies consistent formatting and quality

## Technical Implementation

### AI Models Used
- **Whisper (OpenAI)**: Speech recognition with word-level timestamps
- **GPT-4**: Content analysis and viral moment identification
- **Wan2.2 (5B parameters)**: Text-to-video generation for B-roll
- **MediaPipe**: Face detection for optimal video cropping

### Video Processing Pipeline
1. **Input Processing**: Download/validate source video
2. **Transcription**: Generate accurate transcript with timestamps
3. **AI Analysis**: Identify viral moments using GPT-4
4. **Clip Extraction**: Cut precise video segments
5. **Visual Enhancement**: Crop, enhance, and format for social media
6. **B-roll Integration**: Add AI-generated supplementary footage
7. **Caption Generation**: Create and overlay dynamic captions
8. **Output Optimization**: Export in optimal format for platforms

### Caption System
- **Semantic Highlighting**: Orange text for impactful words
- **Timing Precision**: Word-level synchronization
- **Visual Appeal**: Custom fonts and styling for engagement
- **Readability**: Optimized for mobile viewing

## File Structure

```
autoclipper/
├── clip.py                 # Main processing pipeline
├── run_on_vast.py          # Cloud automation script
├── production_broll.py     # AI B-roll generation system
├── fast_broll.py          # Alternative B-roll implementation
├── testing.py             # Development testing utilities
├── requirements.txt       # Python dependencies
├── README.md              # Basic project documentation
├── cookies.txt            # YouTube authentication
├── keys.txt               # API key storage
├── clips/                 # Generated video clips
├── captions/              # Subtitle files
├── transcripts/           # Text transcriptions
├── broll/                 # AI-generated B-roll videos
├── downloads/             # Source video storage
└── clips_with_broll/      # Final processed videos
```

## Workflow Example

### Input
- YouTube podcast: 2-hour interview
- Local file: 45-minute lecture

### Processing Steps
1. **Download**: Retrieve video in best quality
2. **Transcribe**: Generate full transcript with timestamps
3. **Analyze**: AI identifies 160+ potential moments
4. **Rank**: GPT-4 selects top 20 viral candidates
5. **Extract**: Cut precise video segments
6. **Enhance**: Apply visual improvements and cropping
7. **Generate B-roll**: Create 3-5 supplementary video clips
8. **Caption**: Add dynamic, highlighted subtitles
9. **Composite**: Integrate all elements into final videos

### Output
- 20 polished videos (30-60 seconds each)
- 9:16 aspect ratio for mobile platforms
- Dynamic captions with semantic highlighting
- AI-generated B-roll for visual interest
- Ready for immediate social media posting

## Use Cases

### Content Creators
- Convert long podcasts into multiple short clips
- Maximize content reach across platforms
- Maintain consistent posting schedule

### Educators
- Transform lectures into digestible segments
- Create engaging educational content
- Improve student engagement through visual variety

### Marketers
- Extract key moments from webinars
- Create promotional content from longer presentations
- Generate social media assets at scale

### Podcasters
- Expand audience reach through short-form content
- Create promotional clips for full episodes
- Maintain engagement between episode releases

## Technical Requirements

### Local Development
- Python 3.8+
- FFmpeg for video processing
- CUDA-capable GPU (recommended)
- OpenAI API access
- Hugging Face account

### Cloud Processing
- Vast.ai account and API key
- Sufficient credits for GPU instances
- Stable internet connection for file transfers

## Cost Considerations

### API Usage
- OpenAI GPT-4: ~$0.50-2.00 per video (depending on length)
- Hugging Face model downloads: Free with account
- Vast.ai GPU rental: ~$0.20-0.50 per hour (RTX 5090)

### Time Estimates
- Local processing: 2-4 hours per long video
- Cloud processing: 30-60 minutes per long video
- Setup time: 15-20 minutes for cloud instances

## Quality Optimization

### Content Selection Criteria
- Emotional impact and engagement potential
- Standalone comprehensibility
- Visual interest and variety
- Platform-specific optimization
- Trending topic alignment

### Technical Quality
- 4K output resolution when possible
- Optimized compression for social platforms
- Consistent audio levels
- Professional caption styling
- Smooth B-roll integration

## Future Enhancements

### Planned Features
- Multi-language support
- Custom branding and styling options
- Platform-specific optimization presets
- Automated thumbnail generation
- Social media scheduling integration

### Technical Improvements
- Faster local processing with optimized models
- Advanced B-roll generation with better context
- Improved semantic understanding for clip selection
- Real-time processing capabilities
- Enhanced quality metrics and analytics

## Conclusion

AutoClipper represents a complete solution for modern content creators who need to efficiently transform long-form content into engaging short-form videos. By leveraging cutting-edge AI technologies and automated workflows, it eliminates the time-intensive manual editing process while maintaining high quality and viral potential.

The system is designed for scalability, allowing creators to process multiple videos efficiently while maintaining consistent quality and engagement metrics. Whether used for educational content, marketing materials, or entertainment purposes, AutoClipper provides the tools necessary to succeed in the short-form video landscape.
