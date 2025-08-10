#!/bin/bash
# Vast.ai Setup Script
# Run this on your Vast.ai instance after cloning the repo

echo "🚀 Setting up AutoClipper on Vast.ai..."

# Set environment variables (you'll need to provide these)
export OPENAI_API_KEY="your_openai_key_here"
export ELEVENLABS_API_KEY="your_elevenlabs_key_here"
export MAX_CLIPS=10
export GENERATE_BROLL=true
export PREFER_QUALITY=false

# Install system dependencies
apt-get update
apt-get install -y ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "✅ Setup complete!"
echo "📝 Remember to set your API keys as environment variables"
echo "🎬 Run: python clip.py"
