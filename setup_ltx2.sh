#!/bin/bash
# Setup script for LTX-2 Video Generation
# This script sets up the environment for LTX-2 Fast video generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
VENV_DIR="${ROOT_DIR}/.venv"

echo "=========================================="
echo "🎬 LTX-2 Fast Video Generation Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10+ is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  nvidia-smi not found. CUDA may not be available."
fi
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support (adjust for your CUDA version)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || {
    echo "⚠️  cu121 install failed, trying cu118..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

# Install core dependencies with specific versions for LTX-2
echo "📚 Installing core dependencies..."
pip install diffusers>=0.32.0 transformers>=4.52 accelerate einops
pip install opencv-python imageio[ffmpeg] imageio-ffmpeg
pip install moviepy pillow numpy

# Verify installations
echo ""
echo "🔍 Verifying installations..."
python3 << 'EOF'
import sys

def check_import(module, name=None):
    name = name or module
    try:
        __import__(module)
        print(f"  ✅ {name}")
        return True
    except ImportError as e:
        print(f"  ❌ {name}: {e}")
        return False

all_ok = True
all_ok &= check_import("torch", "PyTorch")
all_ok &= check_import("diffusers", "Diffusers")
all_ok &= check_import("transformers", "Transformers")
all_ok &= check_import("accelerate", "Accelerate")
all_ok &= check_import("einops", "einops")
all_ok &= check_import("cv2", "OpenCV")
all_ok &= check_import("imageio", "imageio")
all_ok &= check_import("moviepy", "moviepy")

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  CUDA not available (CPU only)")
except Exception as e:
    print(f"  ❌ CUDA check failed: {e}")

# Check diffusers version
try:
    import diffusers
    version = diffusers.__version__
    major, minor = map(int, version.split('.')[:2])
    if major > 0 or minor >= 32:
        print(f"  ✅ diffusers {version} (>=0.32.0)")
    else:
        print(f"  ⚠️  diffusers {version} (may need upgrade to >=0.32.0)")
except Exception as e:
    print(f"  ❌ diffusers version check failed: {e}")

if not all_ok:
    print("\n❌ Some dependencies failed to install.")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Dependency verification failed"
    exit 1
fi

# Install OpenAI for GPT analysis
echo ""
echo "🧠 Installing OpenAI client..."
pip install openai>=1.0.0

# Install other project dependencies
echo "📦 Installing remaining dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt 2>/dev/null || echo "⚠️  Some optional dependencies may need manual installation"
fi

# Verify LTX-2 can be imported (will trigger model download on first use)
echo ""
echo "🔍 Testing LTX-2 import..."
python3 << 'EOF'
import warnings
warnings.filterwarnings("ignore")

try:
    from diffusers import LTX2Pipeline
    print("  ✅ LTX2Pipeline import successful")
    print("  📝 Model will download on first generation (~19GB)")
except Exception as e:
    print(f"  ❌ LTX2Pipeline import failed: {e}")
    print("  📝 You may need to upgrade diffusers: pip install -U diffusers")
    exit(1)
EOF

echo ""
echo "=========================================="
echo "✅ LTX-2 Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Login to HuggingFace (required to download LTX-2):"
echo "   huggingface-cli login"
echo "   # Or set HF_TOKEN environment variable"
echo ""
echo "2. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "3. Test the installation:"
echo "   source .venv/bin/activate"
echo "   python video_gen.py"
echo ""
echo "4. Run the B-roll pipeline:"
echo "   python production_broll.py --setup"
echo ""
echo "Environment variables (add to ~/.bashrc or ~/.zshrc):"
echo "   export LTX2_RESOLUTION=480p      # Options: 480p, 720p, 1080p, 4K"
echo "   export LTX2_ASPECT_RATIO=16:9    # Options: 16:9, 9:16"
echo "   export LTX2_FAST_MODE=true       # Faster generation"
echo "   export OPENAI_API_KEY='your-key'"
echo ""
echo "GPU Requirements:"
echo "   - RTX 4090 (24GB): Up to 720p comfortably"
echo "   - RTX 5090 / H100: Up to 4K resolution"
echo "   - LTX-2 Fast mode reduces VRAM usage significantly"
echo ""
echo "Model will be automatically downloaded on first use from:"
echo "   - HuggingFace: Lightricks/LTX-2"
echo ""
echo "If you get CUDA OOM errors:"
echo "   - Lower resolution: export LTX2_RESOLUTION=480p"
echo "   - Enable CPU offload in code: enable_model_cpu_offload=True"
echo ""
