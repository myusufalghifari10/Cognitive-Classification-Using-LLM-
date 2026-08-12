#!/usr/bin/env bash
# 03_eval.sh — SERVE GGUF Q8 (llama-server) + RUN eval.py
# eval.py native llama-server (/props, model "local", response_format, /slots) → TANPA modifikasi.
#
# Run:  bash pod/03_eval.sh           # few-shot (default) + zero-shot
set -euo pipefail

WORK=/workspace
VENV="$WORK/venv"
PROJ="$WORK/Cognitive-Classification-Using-LLM-"
LLAMA="$WORK/llama.cpp"
OUT="$WORK/outputs/qwen36-35b"
GGUF="$OUT/model-q8_0.gguf"
PORT=8889
RES="$PROJ/results/qwen36-35b-a3b-post"

[ -f "$GGUF" ] || {
  echo "✗ GGUF gak ada: $GGUF — run 02_finetune.sh dulu"
  exit 1
}

cd "$PROJ"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "═══════════════════════════════════════════════════════"
echo " [1/3] Start llama-server (port $PORT)"
echo "═══════════════════════════════════════════════════════"
# -ngl 99: offload SEMUA 40 layer ke GPU (Q8 38GB < 48GB ✓)
# -c 24576: context input(~7k) + gen(~16k). Kurangi ke 16384 kalau OOM.
# --jinja: aktifin chat template (eval.py pakai /v1/chat/completions)
SERVER_LOG="/root/llama-server.log"
# --flash-attn on: paksa FA (hemat VRAM KV cache, lebih cepat). Ampere A6000 support penuh.
#   Aman krn 00_setup build pakai GGML_CUDA_FA_ALL_QUANTS=ON → FA jalan utk SEMUA KV type.
# KV cache default FP16 (paling akurat). Kalau OOM di -c 32000 (48GB ketat):
#   tambah  -ctk q8_0 -ctv q8_0  (hemat ~½ KV memory, FA tetap jalan krn FA_ALL_QUANTS).
"$LLAMA/build/bin/llama-server" \
  -m "$GGUF" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 32000 -t 32 \
  --flash-attn on \
  --jinja >"$SERVER_LOG" 2>&1 &
SRV_PID=$!
trap 'echo "  cleanup: kill server $SRV_PID"; kill $SRV_PID 2>/dev/null || true' EXIT

echo "  server PID $SRV_PID (log: $SERVER_LOG)"
echo "  tunggu /props siap ..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/props" >/dev/null 2>&1; then
    echo "  ✓ server siap (${i}s)"
    break
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "✗ server mati — log:"
    tail -20 "$SERVER_LOG"
    exit 1
  fi
  sleep 2
done
curl -s "http://127.0.0.1:$PORT/props" | python -c "import sys,json;d=json.load(sys.stdin);print(f'  model: {d.get(\"model_path\",\"?\")[:60]}  slots: {d.get(\"total_slots\")}')"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " [2/3] EVAL few-shot"
echo "═══════════════════════════════════════════════════════"
# --max-tokens 16384: budget reasoning (5-klasifikasi gak butuh 32k). -c 24576 cukup.
python eval.py --prompt few --host 127.0.0.1 --port "$PORT" \
  --out "$RES" --max-tokens 32000

echo ""
echo "═══════════════════════════════════════════════════════"
echo " [3/3] EVAL zero-shot"
echo "═══════════════════════════════════════════════════════"
python eval.py --prompt zero --host 127.0.0.1 --port "$PORT" \
  --out "$RES" --max-tokens 32000

echo ""
echo "═══════════════════════════════════════════════════════"
echo " ✅ EVAL SELESAI — hasil:"
echo "═══════════════════════════════════════════════════════"
echo "  $RES/"
ls -la "$RES/" 2>/dev/null | grep -E "report_|confusion" || true
echo ""
echo "  Catatan penting:"
echo "  - Hasil auto-save ke /workspace (persist). eval.py punya checkpoint → resume kalau putus."
echo "  - Kalau OOM saat server start: turunin -c ke 16384 di script ini, restart."
echo "  - Kalau banyak PARSE_FAIL/truncation: naikin --max-tokens ke 24576 (+ -c 32768)."
