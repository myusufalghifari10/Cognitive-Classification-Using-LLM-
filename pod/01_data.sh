#!/usr/bin/env bash
# 01_data.sh — CLONE repo + DOWNLOAD base model. Idempotent (skip kalau sudah ada).
# Estimasi: clone ~10 detik + download model ~1-3 menit (RunPod HF cepet)
set -euo pipefail

WORK=/workspace
PROJ="$WORK/Cognitive-Classification-Using-LLM-"
MODELS="$WORK/models"

# ════════════════════════════════════════════════════════════════
# PILIH BASE MODEL:
#   BF16 FULL PRECISION (DEFAULT — 72GB disk, QLoRA 4-bit → ~18GB VRAM, CEPAT):
#     Qwen/Qwen3.6-35B-A3B
#   FP8 (36GB, TAPI Unsloth DISABLE 4-bit → 16bit LoRA → OOM di seq 8192):
#     Qwen/Qwen3.6-35B-A3B-FP8
# ════════════════════════════════════════════════════════════════
MODEL_REPO="Qwen/Qwen3.6-35B-A3B"        # BF16 full precision (72GB) → QLoRA 4-bit
MODEL_NAME="$(basename "$MODEL_REPO")"
MODEL_PATH="$MODELS/$MODEL_NAME"

echo "════════════════════════════════════════════"
echo " [1/2] Clone repo"
echo "════════════════════════════════════════════"
if [ -d "$PROJ/.git" ]; then
  echo "  ✓ udah ada → git pull"
  git -C "$PROJ" pull --ff-only || true
else
  git clone https://github.com/myusufalghifari10/Cognitive-Classification-Using-LLM-.git "$PROJ"
fi
echo "  → repo: $PROJ"
echo "  → dataset: $(wc -l < "$PROJ/data/dataset_sft_sharegpt.jsonl") sampel SFT, \
$(wc -l < "$PROJ/data/test_dataset.csv") baris test"

echo ""
echo "════════════════════════════════════════════"
echo " [2/2] Download base model: $MODEL_REPO"
echo "════════════════════════════════════════════"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
pip install -q -U "huggingface_hub[cli]" 2>/dev/null || pip install -q -U huggingface_hub
mkdir -p "$MODELS"

if [ -f "$MODEL_PATH/config.json" ]; then
  echo "  ✓ udah ada → skip (hapus $MODEL_PATH buat re-download)"
else
  echo "  downloading ke $MODEL_PATH ..."
  # `hf download` = command baru (ganti huggingface-cli yg deprecated).
  #   [verified: github.com/huggingface/huggingface_hub docs/en/guides/cli.md]
  hf download "$MODEL_REPO" --local-dir "$MODEL_PATH"
fi
SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
echo "  → model: $MODEL_PATH ($SIZE)"

echo ""
echo "════════════════════════════════════════════"
echo " ✅ DATA SIAP"
echo "════════════════════════════════════════════"
echo "  Repo:   $PROJ"
echo "  Model:  $MODEL_PATH  ($MODEL_REPO)"
echo "  Catatan MODEL_REPO di atas → dipakai 02_finetune.sh (samakan kalau diganti!)"
echo ""
echo "→ Lanjut: bash pod/02_finetune.sh"
echo "  (atau probe dulu: cd $PROJ && ../venv/bin/python3 train.py --probe --model $MODEL_PATH --qlora)"
