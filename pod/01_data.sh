#!/usr/bin/env bash
# 01_data.sh — CLONE repo + DOWNLOAD base model. Idempotent (skip kalau sudah ada).
# Estimasi: clone ~10 detik + download model ~1-3 menit (RunPod HF cepet)
set -euo pipefail

WORK=/workspace
PROJ="$WORK/Cognitive-Classification-Using-LLM-"
MODELS="$WORK/models"

# ════════════════════════════════════════════════════════════════
# PILIH BASE MODEL (arg $1, default qwen36):
#   qwen36 → Qwen/Qwen3.6-35B-A3B          (BF16 72GB, base asli)
#   kat    → Kwaipilot/KAT-Coder-V2.5-Dev  (BF16 69GB, finetune coding dr Qwen3.6)
# Keduanya BF16 full precision → QLoRA 4-bit ~18GB VRAM. Arsitektur sama (Qwen3_5Moe).
# ════════════════════════════════════════════════════════════════
MODEL_ALIAS="${1:-qwen36}"
case "$MODEL_ALIAS" in
  qwen36) MODEL_REPO="Qwen/Qwen3.6-35B-A3B" ;;
  kat)    MODEL_REPO="Kwaipilot/KAT-Coder-V2.5-Dev" ;;
  *)      echo "✗ alias gak dikenal: $MODEL_ALIAS (pakai: qwen36 | kat)"; exit 1 ;;
esac
MODEL_NAME="$(basename "$MODEL_REPO")"
MODEL_PATH="$MODELS/$MODEL_NAME"
echo "  model: $MODEL_ALIAS → $MODEL_REPO"

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
