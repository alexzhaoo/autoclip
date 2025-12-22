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
  # Build deps are needed for some LightX2V components on fresh images.
  sudo apt-get -o DPkg::Lock::Timeout=300 install -y \
    git ffmpeg python3 python3-venv python3-pip \
    build-essential python3-dev ninja-build
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

# Smoke test: fail early if LightX2V didn't install correctly.
python -c "from lightx2v import LightX2VPipeline; print('✅ LightX2V import OK')"

if [ -f "${ROOT_DIR}/requirements.txt" ]; then
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

# Keep huggingface_hub < 1.0 for compatibility with current transformers/tokenizers.
# The `hf` CLI is shipped with huggingface_hub.
python -m pip install --upgrade "huggingface_hub>=0.24.0,<1.0" hf_transfer

# Newer diffusers PEFT loader expects `transformers.modeling_layers`.
# Keep <4.53 due to vastai-sdk constraints.
python -m pip install --upgrade "transformers>=4.52,<4.53" "diffusers>=0.31.0"
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

# Older `hf download` versions don't support --local-dir-use-symlinks.
HF_LOCAL_DIR_ARGS=()
if hf download --help 2>&1 | grep -q "local-dir-use-symlinks"; then
  HF_LOCAL_DIR_ARGS+=(--local-dir-use-symlinks False)
fi

echo "[setup_wan] Downloading Base Model (Excluding massive noise files)..."
# We exclude the 'high_noise_model' folder to prevent downloading the fine-tuned weights by accident
hf download "${BASE_REPO_ID}" \
  --local-dir "${MODELS_DIR}/Wan2.2-T2V-A14B" \
  "${HF_LOCAL_DIR_ARGS[@]}" \
  --exclude "*high_noise_model*" "*low_noise_model*" "*.git*"

echo "[setup_wan] Downloading Specific Distill LoRAs..."
# We use --include to force download ONLY these files, ignoring the rest of the repo
hf download "${DISTILL_LORA_REPO_ID}" \
  --local-dir "${MODELS_DIR}/loras" \
  "${HF_LOCAL_DIR_ARGS[@]}" \
  "${HIGH_NOISE_LORA}"

hf download "${DISTILL_LORA_REPO_ID}" \
  --local-dir "${MODELS_DIR}/loras" \
  "${HF_LOCAL_DIR_ARGS[@]}" \
  "${LOW_NOISE_LORA}"

# Verify expected files exist
if [ ! -f "${MODELS_DIR}/loras/${HIGH_NOISE_LORA}" ]; then
  echo "[setup_wan] ERROR: High-noise LoRA missing: ${MODELS_DIR}/loras/${HIGH_NOISE_LORA}" >&2
  exit 1
fi
if [ ! -f "${MODELS_DIR}/loras/${LOW_NOISE_LORA}" ]; then
  echo "[setup_wan] ERROR: Low-noise LoRA missing: ${MODELS_DIR}/loras/${LOW_NOISE_LORA}" >&2
  exit 1
fi

echo "[setup_wan] Done. Ready for generation."