#!/usr/bin/env bash
# 01_data.sh — CLONE repo + DOWNLOAD base model. Idempotent (skip kalau sudah ada).
# Estimasi: clone ~10 detik + download model ~1-3 menit (RunPod HF cepet)
set -euo pipefail

WORK=/workspace
PROJ="$WORK/Cognitive-Classification-Using-LLM-"
MODELS="$WORK/models"

# ════════════════════════════════════════════════════════════════
# PILIH BASE MODEL (edit kalau mau ganti):
#   BF16 (PROVEN — kamu udah train 35B di ini sebelumnya):     Qwen/Qwen3.6-35B-A3B
#   FP8  (download 37.5GB lebih kecil, TAPI Unsloth FP8+QLoRA UNTESTED):
#                                                                Qwen/Qwen3.6-35B-A3B-FP8
# ════════════════════════════════════════════════════════════════
MODEL_REPO="Qwen/Qwen3.6-35B-A3B"        # ← ganti ke -FP8 kalau mau coba FP8
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
  huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_PATH"
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
