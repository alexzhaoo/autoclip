# Vast.ai Deployment Guide

## Prerequisites

### Install Vast.ai CLI (on your local machine)
```bash
pip install vastai
```

## Quick Deploy to Vast.ai

### 1. Create Vast.ai Instance
```bash
# Search for instances with GPU and sufficient RAM
vastai search offers 'gpu_name=RTX_4090 gpu_ram>=16 disk_space>=50'

# Rent an instance (replace INSTANCE_ID with actual ID)
vastai create instance INSTANCE_ID --image pytorch/pytorch:latest --disk 50
```

### 2. Clone and Setup
```bash
# SSH into your instance
vastai ssh INSTANCE_ID

# Clone repository (no .env file included - secure!)
git clone https://github.com/alexzhaoo/autoclip.git
cd autoclip

# Set YOUR ACTUAL environment variables BEFORE running
export OPENAI_API_KEY="your_actual_key_here"
export ELEVENLABS_API_KEY="your_actual_key_here"
export GENERATE_BROLL=true
export PREFER_QUALITY=false

# Run setup script
chmod +x setup_vast.sh
./setup_vast.sh
```

### 3. Run Pipeline
```bash
python clip.py
```

## Alternative: Use Vast.ai Environment Variables

When creating your instance, you can set environment variables directly:

```bash
vastai create instance INSTANCE_ID \
  --image pytorch/pytorch:latest \
  --disk 50 \
  --env OPENAI_API_KEY="your_key" \
  --env ELEVENLABS_API_KEY="your_key" \
  --env GENERATE_BROLL="true" \
  --env PREFER_QUALITY="false"
```

## Security Notes
- ✅ `.env` file is gitignored (never committed)
- ✅ API keys set as environment variables on Vast.ai
- ✅ No sensitive data in GitHub repository
- ✅ Template file provided for local development
