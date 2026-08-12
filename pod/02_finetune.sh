#!/usr/bin/env bash
# 02_finetune.sh — FINE-TUNE (QLoRA 4-bit BF16) + MERGE + CONVERT GGUF Q8. SATU PASANG.
#
# Flow:  train.py --skip-gguf  →  LoRA (/workspace, persist)
#                                       →  merged_16bit (/dev/shm, RAM cepat)
#        convert_hf_to_gguf.py → F16     (cepat: cuma copy weight)
#        llama-quantize F16 → Q8         (C++ multi-thread, GB/s — BUKAN numpy 9MB/s!)
#        cp Q8 → /workspace/outputs      (persist buat eval)
#
# Estimasi waktu: train ~3-4 jam + convert ~10-20 menit
# ⚠ Butuh pod RAM ≥180GB (buat /dev/shm nampung merge 66GB + F16 66GB)
set -euo pipefail

WORK=/workspace
VENV="$WORK/venv"
PROJ="$WORK/Cognitive-Classification-Using-LLM-"
LLAMA="$WORK/llama.cpp"
MODELS="$WORK/models"
# OUT di-set di case bawah (parametrik)
SCRATCH="/dev/shm/ft"

# ── samakan dengan 01_data.sh (arg $1, default qwen36) ──
MODEL_ALIAS="${1:-qwen36}"
case "$MODEL_ALIAS" in
  qwen36) MODEL_REPO="Qwen/Qwen3.6-35B-A3B";        OUT="$WORK/outputs/qwen36-35b" ;;
  kat)    MODEL_REPO="Kwaipilot/KAT-Coder-V2.5-Dev"; OUT="$WORK/outputs/kat-coder"  ;;
  *)      echo "✗ alias gak dikenal: $MODEL_ALIAS (pakai: qwen36 | kat)"; exit 1 ;;
esac
MODEL_PATH="$MODELS/$(basename "$MODEL_REPO")"
echo "  finetune target: $MODEL_ALIAS → $MODEL_REPO"

cd "$PROJ"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# OOM fix: expandable_segments kurangi fragmentasi VRAM (OOM margin cuma ~100MB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "═══════════════════════════════════════════════════════"
echo " [0/4] Pre-flight checks"
echo "═══════════════════════════════════════════════════════"
[ -f "$MODEL_PATH/config.json" ] || { echo "✗ model gak ada: $MODEL_PATH — run 01_data.sh dulu"; exit 1; }
[ -x "$LLAMA/build/bin/llama-quantize" ] || { echo "✗ llama.cpp belum dibuild — run 00_setup.sh"; exit 1; }
free -g | awk '/Mem:/{print "  RAM total: "$2" GB"}'
nvidia-smi --query-gpu=memory.total --format=csv,noheader | sed 's/^/  GPU VRAM: /'

# Besarin RAM disk buat merge 66GB + F16 66GB (butuh ~140GB).
# Coba 1: remount /dev/shm gede. Coba 2: mount tmpfs fresh (kalau remount ditolak).
if mount -o remount,size=192G /dev/shm 2>/dev/null; then
  echo "  ✓ /dev/shm remount 192G"
elif mkdir -p /mnt/ft_ram && mount -t tmpfs -o size=200G tmpfs /mnt/ft_ram 2>/dev/null; then
  SCRATCH="/mnt/ft_ram/ft"; echo "  ✓ tmpfs 200G @ /mnt/ft_ram (fallback: remount /dev/shm gagal)"
else
  echo "  ⚠ GAGAL besarin RAM disk — convert bakal lambat/OOM (butuh ~140GB free)"
fi
df -h "$(dirname "$SCRATCH")" 2>/dev/null | awk 'NR==2{print "  scratch: "$2" total, "$4" free"}'

echo ""
echo "═══════════════════════════════════════════════════════"
echo " [1/4] TRAIN (QLoRA 4-bit BF16, batch=1 grad-accum=16, seq=8192)"
echo "═══════════════════════════════════════════════════════"
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"
# BF16 base (72GB) + QLoRA 4-bit → Unsloth quantize ke ~18GB VRAM. MUAT CEPAT di 96GB, seq 8192 utuh.
#   [verified: NVIDIA forum resep 'train BF16 QLoRA' + unsloth MoE 4-bit = 19GB]
python train.py \
  --model "$MODEL_PATH" \
  --data data/dataset_sft_sharegpt.jsonl \
  --out "$OUT" \
  --scratch "$SCRATCH" \
  --qlora \
  --batch 1 --grad-accum 16 \
  --max-seq 8192 \
  --epochs 3 \
  --lora-r 8 \
  --skip-gguf
# → LoRA: $OUT/lora (persist)   merged: $SCRATCH/merged_16bit (RAM)

MERGED="$SCRATCH/merged_16bit"
[ -f "$MERGED/config.json" ] || { echo "✗ merge gagal — cek log train.py"; exit 1; }
echo "  ✓ LoRA: $OUT/lora   |   merged: $(du -sh "$MERGED" | cut -f1)"

# f16+q8 LANGSUNG ke /workspace/$OUT (bukan $SCRATCH).
# /dev/shm cuma 88GB → gak muat merged (67) + f16 (71) sekaligus = 138GB.
# /workspace 250GB cukup (merged di RAM, f16+q8 di disk).
mkdir -p "$OUT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " [2/4] CONVERT → F16 GGUF (copy weight, ~12 menit ke mfs)"
echo "═══════════════════════════════════════════════════════"
F16="$OUT/model.f16.gguf"
python "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outfile "$F16" --outtype f16
echo "  ✓ F16: $(du -sh "$F16" | cut -f1)"
rm -rf "$MERGED"          # free 67GB RAM sebelum quantize

echo ""
echo "═══════════════════════════════════════════════════════"
echo " [3/4] QUANTIZE F16 → Q8_0 (C++ multi-thread, ~10 menit)"
echo "═══════════════════════════════════════════════════════"
Q8="$OUT/model-q8_0.gguf"
"$LLAMA/build/bin/llama-quantize" "$F16" "$Q8" q8_0
rm -f "$F16"
echo "  ✓ Q8_0: $(du -sh "$Q8" | cut -f1)"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " [4/4] OUTPUT PERSIST (tahan pod restart)"
echo "═══════════════════════════════════════════════════════"
echo "  ✓ $Q8  ($(du -sh "$Q8" | cut -f1))"
echo "  ✓ $OUT/lora/  (adapter, persist)"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " ✅ FINE-TUNE + Q8 SELESAI"
echo "═══════════════════════════════════════════════════════"
echo "  GGUF:  $OUT/model-q8_0.gguf"
echo "  LoRA:  $OUT/lora/"
echo ""
echo "→ Lanjut: bash pod/03_eval.sh"
