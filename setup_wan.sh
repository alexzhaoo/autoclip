#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
MODELS_DIR="${ROOT_DIR}/models"
LIGHTX2V_DIR="${ROOT_DIR}/LightX2V"

# Repo IDs
BASE_REPO_ID="Wan-AI/Wan2.2-T2V-A14B"
DISTILL_LORA_REPO_ID="lightx2v/Wan2.2-Distill-Loras"

# File Names
HIGH_NOISE_LORA="wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"
LOW_NOISE_LORA="wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors"

echo "[setup_wan] root: ${ROOT_DIR}"
mkdir -p "${MODELS_DIR}/loras"

# -------------------------
# 1. System Dependencies
# -------------------------
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get -o DPkg::Lock::Timeout=300 update
  sudo apt-get -o DPkg::Lock::Timeout=300 install -y \
    git ffmpeg python3 python3-venv python3-pip \
    build-essential python3-dev ninja-build
fi

# -------------------------
# 2. DOWNLOAD PHASE (Use Modern Tooling)
# -------------------------
# We create a temporary venv just for downloading to avoid conflict with the runtime env
echo "[setup_wan] Setting up temporary download environment..."
if [ -d "${ROOT_DIR}/.venv_dl" ]; then rm -rf "${ROOT_DIR}/.venv_dl"; fi
python3 -m venv "${ROOT_DIR}/.venv_dl"
source "${ROOT_DIR}/.venv_dl/bin/activate"

# Install latest CLI for fast, correct downloads
pip install --upgrade "huggingface_hub[cli]>=0.27.0" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "[setup_wan] Downloading Base Model (Excluding massive noise files)..."
huggingface-cli download "${BASE_REPO_ID}" \
  --local-dir "${MODELS_DIR}/Wan2.2-T2V-A14B" \
  --exclude "*high_noise_model*" "*low_noise_model*" "*.git*"

echo "[setup_wan] Downloading Specific Distill LoRAs..."
huggingface-cli download "${DISTILL_LORA_REPO_ID}" "${HIGH_NOISE_LORA}" \
  --local-dir "${MODELS_DIR}/loras"

huggingface-cli download "${DISTILL_LORA_REPO_ID}" "${LOW_NOISE_LORA}" \
  --local-dir "${MODELS_DIR}/loras"

# Deactivate and clean up download env
deactivate
rm -rf "${ROOT_DIR}/.venv_dl"

# -------------------------
# 3. RUNTIME ENVIRONMENT (The Code Setup)
# -------------------------
echo "[setup_wan] Setting up Runtime Environment..."

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  python3 -m venv "${ROOT_DIR}/.venv"
fi
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip

# Install LightX2V Repo
if [ ! -d "${LIGHTX2V_DIR}" ]; then
  git clone --depth 1 https://github.com/ModelTC/LightX2V.git "${LIGHTX2V_DIR}"
fi
python -m pip install -v "${LIGHTX2V_DIR}"

if [ -f "${ROOT_DIR}/requirements.txt" ]; then
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

# Install the SPECIFIC versions needed for runtime compatibility
# We force these versions to satisfy vastai-sdk and LightX2V constraints
echo "[setup_wan] Installing compatible runtime libraries..."
pip install "transformers>=4.52,<4.53" "diffusers>=0.31.0" "tokenizers>=0.19,<0.20" "huggingface_hub<1.0"

# Smoke test
python -c "from lightx2v import LightX2VPipeline; print('✅ LightX2V Runtime Ready')"

echo "[setup_wan] Done. Models downloaded and Runtime Env created."