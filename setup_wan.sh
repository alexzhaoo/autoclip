#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
MODELS_DIR="${ROOT_DIR}/models"
LIGHTX2V_DIR="${ROOT_DIR}/LightX2V"

# Repo IDs
BASE_REPO_ID="Wan-AI/Wan2.2-T2V-A14B"
DISTILL_LORA_REPO_ID="lightx2v/Wan2.2-Distill-Loras"

# File Names (Exact Match)
HIGH_NOISE_LORA="wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"
LOW_NOISE_LORA="wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors"

echo "[setup_wan] root: ${ROOT_DIR}"
mkdir -p "${MODELS_DIR}/loras"

# -------------------------
# 1. System Dependencies
# -------------------------
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get -o DPkg::Lock::Timeout=300 update
  sudo apt-get -o DPkg::Lock::Timeout=300 install -y git ffmpeg python3 python3-venv python3-pip
fi

# -------------------------
# 2. Python Environment
# -------------------------
if [ ! -d "${ROOT_DIR}/.venv" ]; then
  python3 -m venv "${ROOT_DIR}/.venv"
fi
# shellcheck disable=SC1091
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip

# -------------------------
# 3. Install LightX2V & Modern Hugging Face
# -------------------------
if [ ! -d "${LIGHTX2V_DIR}" ]; then
  git clone --depth 1 https://github.com/ModelTC/LightX2V.git "${LIGHTX2V_DIR}"
fi
python -m pip install -v "${LIGHTX2V_DIR}"

if [ -f "${ROOT_DIR}/requirements.txt" ]; then
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

# IMPORTANT: 'hf' command requires a newer version. 
# The [cli] extra is deprecated in newer versions; the CLI is now core.
python -m pip install --upgrade "huggingface_hub>=0.27.0" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# -------------------------
# 4. Download Weights (The Modern Way)
# -------------------------
if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' command not found. Ensure pip install succeeded." >&2
  exit 1
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "[setup_wan] Downloading Base Model (Excluding massive noise files)..."
# We exclude the 'high_noise_model' folder to prevent downloading the fine-tuned weights by accident
hf download "${BASE_REPO_ID}" \
  --local-dir "${MODELS_DIR}/Wan2.2-T2V-A14B" \
  --exclude "*high_noise_model*" "*low_noise_model*" "*.git*"

echo "[setup_wan] Downloading Specific Distill LoRAs..."
# We use --include to force download ONLY these files, ignoring the rest of the repo
hf download "${DISTILL_LORA_REPO_ID}" \
  --include "${HIGH_NOISE_LORA}" \
  --local-dir "${MODELS_DIR}/loras"

hf download "${DISTILL_LORA_REPO_ID}" \
  --include "${LOW_NOISE_LORA}" \
  --local-dir "${MODELS_DIR}/loras"

echo "[setup_wan] Done. Ready for generation."