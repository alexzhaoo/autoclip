#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
MODELS_DIR="${ROOT_DIR}/models"
LIGHTX2V_DIR="${ROOT_DIR}/LightX2V"

# Repo IDs
BASE_REPO_ID="Wan-AI/Wan2.2-T2V-A14B"
DISTILL_LORA_REPO_ID="lightx2v/Wan2.2-Distill-Loras"

# File Names (Files are at the ROOT of the repo)
HIGH_NOISE_LORA="wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"
LOW_NOISE_LORA="wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors"

echo "[setup_wan] root: ${ROOT_DIR}"
mkdir -p "${MODELS_DIR}/loras"

# -------------------------
# System deps
# -------------------------
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get -o DPkg::Lock::Timeout=300 update
  sudo apt-get -o DPkg::Lock::Timeout=300 install -y git ffmpeg python3 python3-venv python3-pip
fi

# -------------------------
# Python env
# -------------------------
if [ ! -d "${ROOT_DIR}/.venv" ]; then
  python3 -m venv "${ROOT_DIR}/.venv"
fi
# shellcheck disable=SC1091
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip

# -------------------------
# Install LightX2V
# -------------------------
if [ ! -d "${LIGHTX2V_DIR}" ]; then
  git clone --depth 1 https://github.com/ModelTC/LightX2V.git "${LIGHTX2V_DIR}"
fi
python -m pip install -v "${LIGHTX2V_DIR}"

if [ -f "${ROOT_DIR}/requirements.txt" ]; then
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

python -m pip install --upgrade "huggingface_hub>=0.24.0" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# -------------------------
# Download weights
# -------------------------
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found" >&2
  exit 1
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# 1. Base Model Download (Strict Filters)
# We exclude "high_noise_model" folders to stop that 10GB download
echo "[setup_wan] Downloading base model: ${BASE_REPO_ID}"
huggingface-cli download "${BASE_REPO_ID}" \
  --local-dir "${MODELS_DIR}/Wan2.2-T2V-A14B" \
  --local-dir-use-symlinks False \
  --exclude "*high_noise_model*" "*low_noise_model*" "*.git*"

# 2. LoRA Download (Direct Target)
# We target the files exactly so it doesn't download the whole repo
echo "[setup_wan] Downloading High Noise LoRA..."
huggingface-cli download "${DISTILL_LORA_REPO_ID}" "${HIGH_NOISE_LORA}" \
  --local-dir "${MODELS_DIR}/loras" \
  --local-dir-use-symlinks False

echo "[setup_wan] Downloading Low Noise LoRA..."
huggingface-cli download "${DISTILL_LORA_REPO_ID}" "${LOW_NOISE_LORA}" \
  --local-dir "${MODELS_DIR}/loras" \
  --local-dir-use-symlinks False

echo "[setup_wan] Done. Models layout:" 
echo "  - ${MODELS_DIR}/Wan2.2-T2V-A14B"
echo "  - ${MODELS_DIR}/loras/${HIGH_NOISE_LORA}"
echo "  - ${MODELS_DIR}/loras/${LOW_NOISE_LORA}"