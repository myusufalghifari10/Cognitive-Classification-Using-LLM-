#!/usr/bin/env bash
# 00_setup.sh — INSTALL SEMUA di fresh pod. Run SEKALI per pod start.
# Output: /workspace/venv (unsloth + fast-path libs) + /workspace/llama.cpp (CUDA build)
# Estimasi: ~10-15 menit
#
# RESEP TERVERIFIKASI (pod 3aa922cc1fbf, RTX A6000, 2026-08):
#   • torch 2.11.0+cu13 (CUDA build) — via unsloth tanpa --torch-backend
#   • causal-conv1d (precompiled wheel cu13+torch2.11) → fast path GDN aktif
#   • flash-linear-attention (fla) → fast path linear-attn aktif
#   • llama.cpp build arch 86 (Ampere)
set -euo pipefail

WORK=/workspace
VENV="$WORK/venv"
LLAMA="$WORK/llama.cpp"
PROJ="$WORK/Cognitive-Classification-Using-LLM-"
export UV_CONCURRENT_DOWNLOADS=4

# nvcc di PATH + CUDA_HOME (buat causal-conv1d/fla source-build fallback kalau precompiled wheel gak ada)
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# ── Deteksi GPU arch buat llama.cpp build (sm_XX) ──
#   Ampere:  A100=80, A6000/A40/3090=86
#   Ada:     L4/L40/4090=89
#   Hopper:  H100=90
#   Blackwell: RTX 5090/5080/PRO 6000=120, B100/B200=100
GPU_ARCH="${GPU_ARCH:-}"
if [ -z "$GPU_ARCH" ]; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  case "$CC" in
    8.0)  GPU_ARCH=80 ;;
    8.6)  GPU_ARCH=86 ;;
    8.9)  GPU_ARCH=89 ;;
    9.0)  GPU_ARCH=90 ;;
    10.0) GPU_ARCH=100 ;;
    12.0) GPU_ARCH=120 ;;   # Blackwell RTX 50/PRO 6000
    *)    GPU_ARCH=86; echo "  ⚠ compute_cap '$CC' gak dikenal → default arch 86" ;;
  esac
fi
echo "  GPU arch terdeteksi: sm_$GPU_ARCH (compute_cap=$CC)"

echo "════════════════════════════════════════════"
echo " [0/5] apt deps"
echo "════════════════════════════════════════════"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake build-essential python3-venv python3-pip python3-dev curl > /dev/null

echo "════════════════════════════════════════════"
echo " [1/5] llama.cpp (CUDA build, arch 86 = Ampere A6000)"
echo "════════════════════════════════════════════"
# persist di mfs biar gak rebuild tiap pod migration
if [ ! -d "$LLAMA/.git" ]; then
  rm -rf "$LLAMA"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA"
fi
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  # arch auto-detect (sm_XX). Blackwell RTX PRO 6000=120, A6000=86, A100=80, H100=90.
  # GGML_CUDA_FA_ALL_QUANTS=ON: FlashAttention CUDA kernel dukung SEMUA KV cache quant (q8_0/q4_0 dll).
  #   TANPA ini: --flash-attn + KV quant → SILENT fallback ke CPU attention = "extremely slow".
  #   [verified: llama.cpp#24485 + build.md GGML_CUDA_FA_ALL_QUANTS table]
  # Cost: build ~2x lebih lama (~8 min), one-time.
  cmake -S "$LLAMA" -B "$LLAMA/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 || \
      cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build "$LLAMA/build" --config Release -j --target llama-server llama-quantize 2>&1 | tail -2
fi

echo "════════════════════════════════════════════"
echo " [2/5] uv venv → /workspace/venv"
echo "════════════════════════════════════════════"
pip install -q uv
# venv sehat? (mfs kadang break symlink) → kalau rusak recreate
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import unsloth" 2>/dev/null; then
  rm -rf "$VENV"
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q --upgrade pip

echo "════════════════════════════════════════════"
echo " [3/5] Unsloth + torch cu128 (PINNED — match driver 12.8)"
echo "════════════════════════════════════════════"
# ⚠ CRITICAL: --torch-backend=cu128 di SETIAP command.
#   Alasan: nvcc 12.8 (pod) = driver 12.8. torch cu128 match → causal-conv1d/fla source-build OK.
#   Kalau tanpa pin → uv balikin torch cu13/2.13 → ABI mismatch di causal-conv1d build.
#   Blackwell sm_120 support penuh di cu128 wheels (uv issue #14742).
#   [verified: pod sebelumnya, --torch-backend=cu128 = working recipe]
uv pip install --torch-backend=cu128 unsloth

echo "════════════════════════════════════════════"
echo " [4/5] Fast-path libs: causal-conv1d + flash-linear-attention"
echo "════════════════════════════════════════════"
# Qwen3.6 GDN/linear-attn butuh ini buat fast path. Tanpa → fallback torch = 3-4x lebih lambat.
# causal-conv1d: source build (nvcc 12.8 = torch cu128 → ABI match). Pin cu128 biar gak thrash.
uv pip install --torch-backend=cu128 causal-conv1d --no-build-isolation || echo "  ⚠ causal-conv1d gagal — training bakal slow-path"
uv pip install --torch-backend=cu128 flash-linear-attention            || echo "  ⚠ fla gagal — training bakal slow-path"

echo "════════════════════════════════════════════"
echo " [5/5] eval.py + convert deps"
echo "════════════════════════════════════════════"
uv pip install -q requests scikit-learn matplotlib seaborn gguf sentencepiece

echo ""
echo "════════════════════════════════════════════"
echo " ✅ VERIFY — semua harus ✓"
echo "════════════════════════════════════════════"
ok=1
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && \
  python -c "import torch; print(f'  ✓ torch {torch.__version__} | CUDA=True | {torch.cuda.get_device_name(0)}')" \
  || { echo "  ✗ torch CUDA=False (CPU build!) — fix: uv pip install --force-reinstall torch torchvision torchaudio"; ok=0; }
python -c "import causal_conv1d" 2>/dev/null && echo "  ✓ causal-conv1d (fast path GDN)" || { echo "  ✗ causal-conv1d missing → training LAMBAT"; ok=0; }
python -c "import fla" 2>/dev/null && echo "  ✓ flash-linear-attention (fast path)" || { echo "  ✗ fla missing → training LAMBAT"; ok=0; }
python -c "import unsloth; print(f'  ✓ unsloth {unsloth.__version__}')" 2>/dev/null || { echo "  ✗ unsloth missing"; ok=0; }
[ -x "$LLAMA/build/bin/llama-server" ] && echo "  ✓ llama-server + llama-quantize" || { echo "  ✗ llama.cpp build"; ok=0; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
echo ""
[ $ok -eq 1 ] && echo "  → Semua ✓. Lanjut: bash pod/01_data.sh" || echo "  → Ada ✗ di atas. Fix dulu sebelum lanjut (training/probe bakal error)."
