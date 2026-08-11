#!/usr/bin/env bash
# 00_setup.sh — INSTALL SEMUA di fresh pod. Run SEKALI per pod start.
# Output: /workspace/venv (unsloth + eval deps) + /workspace/llama.cpp (build CUDA)
# Estimasi: ~10-15 menit (apt + build llama.cpp + pip unsloth)
set -euo pipefail

WORK=/workspace
VENV="$WORK/venv"
LLAMA="$WORK/llama.cpp"
PROJ="$WORK/Cognitive-Classification-Using-LLM-"
export UV_CONCURRENT_DOWNLOADS=4          # cegah stall (RunPod throttle >4 parallel)

echo "════════════════════════════════════════════"
echo " [0/5] apt deps (cmake, git, build-tools)"
echo "════════════════════════════════════════════"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake build-essential python3-venv python3-pip python3-dev curl > /dev/null

echo "════════════════════════════════════════════"
echo " [1/5] llama.cpp (CUDA build) → /workspace/llama.cpp"
echo "════════════════════════════════════════════"
# persist di mfs biar gak rebuild tiap pod migration
if [ ! -d "$LLAMA/.git" ]; then
  rm -rf "$LLAMA"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA"
fi
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  # GGML_CUDA=ON: butuh nvcc. RunPod PyTorch image biasanya udah ada.
  # CMAKE_CUDA_ARCHITECTURES=86 = Ampere (RTX A6000/A40/A4000). Pin 1 arch = build cepat.
  #   A100=80, H100=90, L40=89, RTX 4090=89, RTX 3090=86. Ganti sesuai GPU kamu.
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 || \
    cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build "$LLAMA/build" --config Release -j --target llama-server llama-quantize 2>&1 | tail -2
fi
echo "  ✓ llama-server:  $LLAMA/build/bin/llama-server"
echo "  ✓ llama-quantize: $LLAMA/build/bin/llama-quantize"

echo "════════════════════════════════════════════"
echo " [2/5] uv venv → /workspace/venv"
echo "════════════════════════════════════════════"
pip install -q uv
# cek: venv masih sehat (mfs kadang break symlink)? kalau rusak → recreate
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import unsloth" 2>/dev/null; then
  rm -rf "$VENV"
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q --upgrade pip

echo "════════════════════════════════════════════"
echo " [3/5] Unsloth (--torch-backend=auto)"
echo "════════════════════════════════════════════"
uv pip install -q "unsloth" --torch-backend=auto

echo "════════════════════════════════════════════"
echo " [4/5] causal-conv1d (--no-build-isolation)"
echo "════════════════════════════════════════════"
uv pip install -q causal-conv1d --no-build-isolation || echo "  ⚠ causal-conv1d gagal (OK kalau arch gak butuh)"

echo "════════════════════════════════════════════"
echo " [5/5] eval.py + convert deps"
echo "════════════════════════════════════════════"
uv pip install -q requests scikit-learn matplotlib seaborn gguf sentencepiece

echo ""
echo "════════════════════════════════════════════"
echo " ✅ SETUP SELESAI — verifikasi:"
echo "════════════════════════════════════════════"
"$LLAMA/build/bin/llama-server" --version 2>&1 | head -1 || true
python -c "import unsloth, transformers, gguf; print(f'unsloth {unsloth.__version__} | transformers {transformers.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "→ Lanjut: bash pod/01_data.sh"
