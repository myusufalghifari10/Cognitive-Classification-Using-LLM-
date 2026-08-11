# RUNBOOK — Deploy Qwen3.6-35B-A3B Fine-tune + Eval di RunPod (48GB GPU)

End-to-end: **fresh pod → install → fine-tune QLoRA → GGUF Q8 → eval.py**.
Total ~4-5 jam (dominan training). Setup data < 20 menit.

---

## 0. Buat Pod — spec yang benar (HEMAT)

| Item | Rekomendasi | Alasan |
|---|---|---|
| GPU VRAM | **48GB** (A40 / L40 / A6000) | Q8 38GB + KV cache muat |
| **System RAM** | **≥ 180GB** ⚠ | /dev/shm nampung merge 66GB + F16 66GB untuk convert cepat |
| Disk | 80GB container | overlay `/` buat venv |
| Network Volume | **attach /workspace yang sama** | LoRA + GGUF + repo persist antar pod |
| Image | RunPod PyTorch (CUDA 12.x) | udah ada nvcc buat build llama.cpp |

⚠ **RAM ≥180GB itu WAJIB** buat convert cepat. Kalau pod cuma 96GB RAM → convert lambat/OOM (fallback ke mfs = 1.43 MB/s). Pilih pod dengan RAM gede.

---

## 1. Eksekusi (urutan, copy-paste)

```bash
# di pod (sebagai root):

# [0] INSTALL semua — ~10-15 menit (apt + build llama.cpp + pip unsloth)
cd /workspace
git clone https://github.com/myusufalghifari10/Cognitive-Classification-Using-LLM-.git
cd Cognitive-Classification-Using-LLM-   # repo name ada trailing dash "-"
bash pod/00_setup.sh

# [1] DATA — clone (lagi biar idempotent) + download model — ~3 menit
bash pod/01_data.sh

# (opsional) PROBE dulu — cek VRAM/token sebelum training mahal:
cd /workspace/Cognitive-Classification-Using-LLM-
/workspace/venv/bin/python3 train.py --probe --model /workspace/models/Qwen3.6-35B-A3B --qlora

# [2] FINE-TUNE + Q8 — ~3-4 jam train + ~15 menit convert (SATU PERINTAH)
bash pod/02_finetune.sh

# [3] EVAL — serve + run eval.py few & zero shot
bash pod/03_eval.sh
```

### Verifikasi tiap step
| Step | Sukses kalau... |
|---|---|
| 00 | `unsloth X.Y | transformers Z.W` + `llama-server version` tercetak |
| 01 | `$MODELS/Qwen3.6-35B-A3B/config.json` ada + repo clone |
| 02 | `$OUT/model-q8_0.gguf` (~38GB) + `$OUT/lora/` ada |
| 03 | `results/qwen36-35b-a3b-post/report_fewshot.txt` ada (accuracy) |

---

## 2. Path & Artifact

```
/workspace/                                    # mfs — PERSIST antar pod
├── Cognitive-Classification-Using-LLM-/       # repo (kode + data + prompt) ⚠ trailing dash
├── venv/                                      # python env (recreate kalau symlink break)
├── llama.cpp/                                 # + build/ (persist, skip rebuild)
├── models/Qwen3.6-35B-A3B/                    # base model BF16 (~66GB)
├── outputs/qwen36-35b/
│   ├── lora/                                  # ⭐ LoRA adapter (irreplaceable — 3-4 jam training)
│   └── model-q8_0.gguf                        # ⭐ GGUF Q8 buat eval (~38GB)
└── Cognitive-.../results/qwen36-35b-a3b-post/ # hasil eval (persist)
/dev/shm/                                      # RAM — HILANG tiap restart (temp only)
```

---

## 3. Recovery kalau pod restart/migrate

Pod migration = overlay `/` reset (venv hilang). `/workspace` (mfs) **tetap**.

```bash
# venv & /dev/shm hilang, tapi repo/model/lora/gguf aman di /workspace.
# Recovery = reinstall venv + (kalau perlu) re-convert. TIDAK re-train (lora persist).

bash pod/00_setup.sh          # rebuild venv (lama: build llama.cpp skip kalau ada)
# Kalau LoRA + GGUF udah ada di /workspace/outputs → langsung:
bash pod/03_eval.sh           # serve GGUF + eval
# Kalau GGUF hilang tapi LoRA ada → re-convert aja (skip train):
#   (edit 02_finetune.sh: comment bagian "TRAIN", uncomment jalur convert-only)
```

---

## 4. Key Decisions & Catatan

### Kenapa QLoRA (4-bit), bukan FP8/BF16 langsung?
- **BF16 (66GB) & FP8 (37.5GB) GAK muat 48GB buat TRAINING** (base + LoRA + gradient + aktivasi > 48GB).
- QLoRA 4-bit: base ~18GB + MoE 3B active (aktivasi kecil) → **fit 48GB**.
- Jadi `--qlora` wajib di 48GB, apapun sumber download-nya.

### FP8 vs BF16 sebagai sumber download?
- Default script: **BF16** (`Qwen/Qwen3.6-35B-A3B`) — **PROVEN**, kamu udah train 35B di sini.
- FP8 (`-FP8`, 37.5GB) download lebih kecil, **TAPI** Unsloth load FP8+QLoRA = **UNTESTED** (bisa gagal load → buang waktu). Mau coba? edit `MODEL_REPO` di `01_data.sh` + `02_finetune.sh`.
- Download diff cuma ~1-2 menit di RunPod → **BF16 lebih aman, worth it**.

### Kenapa convert manual (F16 → Q8), bukan train.py auto?
- train.py auto (`save_pretrained_gguf`) pakai path numpy → **9 MB/s, ~60 menit** 🐌
- Manual: `convert_hf_to_gguf --outtype f16` (cepat) + `llama-quantize` (C++ multi-thread, GB/s) → **~10-15 menit**.
- Makanya `02_finetune.sh` pakai `--skip-gguf` lalu convert manual.

### eval.py TANPA modifikasi
- eval.py native llama-server (`/props`, `model:"local"`, `response_format` llama-style, `/slots erase`).
- llama-server gak validasi nama model → `model:"local"` langsung diterima.
- Jadi gak perlu utak-atik eval.py (beda sama vLLM yang butuh modifikasi).

### Sampling: default server
- eval.py gak override temp/top_p/top_k → pakai default llama-server. Konsisten buat pre vs post comparison.

---

## 5. Troubleshooting

| Gejala | Solusi |
|---||
| `CUDA out of memory` saat train | turunin `--max-seq` ke 6144, atau batch tetap 1 |
| convert OOM (`No space left on /dev/shm`) | pod RAM < 180GB. Besarin: `mount -o remount,size=XXXG /dev/shm`, atau fallback convert di /workspace (lambat) |
| llama-server OOM saat serve | turunin `-c` ke 16384 |
| Banyak `PARSE_FAIL` di eval | naikin `--max-tokens` ke 24576 (+ `-c 32768` di serve) |
| venv broken setelah migrate | `bash pod/00_setup.sh` (auto-detect & recreate) |
| training LAMBAT (>200 s/it) | fast path OFF → `python -c "import causal_conv1d"`. Kalau gagal: `uv pip install causal-conv1d --no-build-isolation && uv pip install flash-linear-attention`. **Akar: torch CPU** (JANGAN `--torch-backend=auto`; no-backend = cu13) |
| `torch==X+cpu` / CUDA False | `uv pip install --force-reinstall torch torchvision torchaudio` (default PyPI = cu13 CUDA build) |
| `llama-server: command not found` | build belum selesai/ambil target salah → `cmake --build $LLAMA/build --target llama-server` |
