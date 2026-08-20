# Cognitive Classification Using LLM

Benchmark 19 model LLM untuk klasifikasi **Cognitive Presence** 5-kelas pada 120 postingan forum diskusi mahasiswa. Model dijalankan via **llama.cpp (`llama-server`)** dengan prompt decontaminated (few-shot & zero-shot). DeepSeek-V4-Flash via API (OpenCode Zen). 2 model di antaranya di-*fine-tune* via **Unsloth** (Qwen3.5-9B-Fable: LoRA 16-bit; Qwen3.6-35B-A3B: QLoRA 4-bit).

Repo ini melanjutkan riset `cognitive-classification` (IndoBERT) dengan **LLM generik sebagai baseline**, dan menjadi fondasi untuk *Prescriptive Learning Analytics dengan Agentic AI*.

---

## Benchmark Results

> Semua run pakai `seed=42`, prompt decontaminated (tidak ada teks test-set di prompt). 120 sampel test. Diurutkan by **Accuracy**. Kolom **Acc** & **Weighted-F1** di-*highlight*.

### Pre-Finetune — Few-Shot (definisi + contoh + aturan, `system_few_shot.md`)

Latensi sengaja tidak dilaporkan — campuran llama-server lokal (hardware berbeda-beda) dan API (queue server) membuat perbandingan tidak adil.

| Model | Size | **Acc** | Macro-F1 | **Weighted-F1** |
|---|---|---|---|---|
| BigBang | 35B A3B | **0.9500** | 0.9594 | **0.9520** |
| Qwen3.8-27B-Fable | 27B dense | **0.9500** | 0.9225 | **0.9500** |
| DeepSeek-V4-Flash ¹ | API | **0.9500** | 0.9172 | **0.9513** |
| Qwen3.6-35B-A3B | 35B A3B | **0.9417** | 0.9290 | **0.9417** |
| DeepSeek-V4-Flash ¹ ² | API | **0.9417** | 0.9080 | **0.9427** |
| Qwen3.6-27B-Fable | 27B dense | **0.9333** | 0.9085 | **0.9336** |
| KAT-Coder2.5 | 35B A3B | **0.9250** | 0.9332 | **0.9262** |
| Ornith1.5-35B-A3B | 35B A3B | **0.9250** | 0.9066 | **0.9265** |
| Qwen3.6-35B-A3B-Fable | 35B A3B | **0.9250** | 0.9063 | **0.9233** |
| Qwen3.8-27B-GAIN | 27B dense | **0.9083** | 0.9029 | **0.9084** |
| Qwen3.8-27B | 27B dense | **0.9083** | 0.8801 | **0.9103** |
| Gemma4-26B-A4B-IT | 26B A4B | **0.9000** | 0.8557 | **0.9036** |
| Qwen3.6-40B | 40B dense | **0.8917** | 0.8604 | **0.8905** |
| Qwen3.5-9B-Fable | 9B | **0.8833** | 0.8288 | **0.8843** |
| Qwen3.6-35B-A3B-DSV4Pro | 35B A3B | **0.8750** | 0.8434 | **0.8804** |
| Muse-Glimmer | 30B dense | **0.8583** | 0.8294 | **0.8625** |
| Nanbeige4.2-3B | 4B | **0.8000** | 0.7577 | **0.8108** |
| Nemotron | 30B A3B | **0.7881** | 0.7026 | **0.7858** |
| Qwythos | 9B | **0.7863** | 0.7458 | **0.7896** |
| GLM-4.7-Flash | 30B A3B | **0.7750** | 0.7154 | **0.7870** |

¹ DeepSeek-V4-Flash dijalankan via API OpenCode Zen (`opencode.ai/zen/go/v1`), bukan llama-server lokal; thinking default (effort `high` per dok DeepSeek).
² Thinking `max` (`reasoning_effort=max`): akurasi turun 0.9500 → 0.9417 — overthinking; analisis biaya-latensi lengkap di bagian Cost & Speed.

### Pre-Finetune — Zero-Shot (hanya definisi kelas, `system_zero_shot.md`)

| Model | Size | **Acc** | Macro-F1 | **Weighted-F1** |
|---|---|---|---|---|
| Qwen3.6-35B-A3B-Fable | 35B A3B | **0.7750** | 0.7094 | **0.7784** |
| Qwen3.6-35B-A3B | 35B A3B | **0.7667** | 0.7062 | **0.7684** |
| Qwen3.5-9B-Fable | 9B | **0.7583** | 0.6794 | **0.7638** |
| KAT-Coder2.5 | 35B A3B | **0.7479** | 0.6788 | **0.7521** |
| BigBang | 35B A3B | **0.7059** | 0.6549 | **0.7162** |

### Post-Finetune — Few-Shot

Fine-tune 3 epoch di Unsloth pada dataset SFT (1080 sampel). Lihat detail metode per model.

| Model | Size | Method | **Acc** | Macro-F1 | **Weighted-F1** |
|---|---|---|---|---|---|
| Qwen3.5-9B-Fable | 9B | LoRA 16-bit | **0.9250** | 0.9165 | **0.9260** |
| Qwen3.6-35B-A3B | 35B A3B | QLoRA 4-bit | **0.9250** | 0.9123 | **0.9244** |

### Post-Finetune — Zero-Shot

| Model | Size | Method | **Acc** | Macro-F1 | **Weighted-F1** |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 35B A3B | QLoRA 4-bit | **0.7917** | 0.7722 | **0.7877** |
| Qwen3.5-9B-Fable | 9B | LoRA 16-bit | _pending_ | — | — |

**Temuan kunci (Pre vs Post):**
- **9B-Fable (LoRA 16-bit)** — Few-Shot 0.8833 → **0.9250** (↑3.7 pp). Headroom + gradien bersih = naik.
- **35B-A3B (QLoRA 4-bit)** — Few-Shot 0.9417 → **0.9250** (↓1.7 pp); Zero-Shot 0.7667 → **0.7917** (↑2.5 pp). Base udah di langit-langit + noise QLoRA → few-shot turun, tapi zero-shot naik (prompt minim paling diuntungkan).

---

## Label (5 kelas CoI — Linear Algebra "Ruang Vektor")

| Kode | Kelas | Makna | Support |
|---|---|---|---|
| C0 | Bukan Kehadiran Kognitif | sapaan, terima kasih, logistik, tanpa substansi materi | 29 |
| C1 | Pemantik (Triggering) | bertanya, kebingungan, merumuskan masalah | 18 |
| C2 | Eksplorasi (Exploration) | bertukar info/opini, definisi, contoh, setuju + alasan | 20 |
| C3 | Integrasi (Integration) | sintesis ide, refleksi menamai konsep, menyusun pembuktian | 48 |
| C4 | Resolusi (Resolution) | simpulan final, solusi lengkap, presentasi hasil kelompok | 5 |

---

## Struktur

```
Cognitive-Classification-Using-LLM/
├── data/
│   ├── training_dataset.csv             # train set (960)
│   ├── validation_dataset.csv           # val set (120)
│   ├── test_dataset.csv                 # test set (120)
│   ├── trainval_dataset.csv             # train+val gabungan (1080)
│   ├── teacher_reasoning.jsonl          # reasoning train+val (1080, Mode Guru)
│   ├── teacher_reasoning_validation.jsonl
│   └── dataset_sft_sharegpt.jsonl       # dataset SFT ShareGPT (1080) siap fine-tune
├── prompts/
│   ├── system_few_shot.md / system_zero_shot.md   # prompt benchmark
│   ├── guru_system.txt                  # rubrik + Mode Guru (distilasi reasoning)
│   ├── test_system.txt                  # system prompt target SFT
│   ├── user.md
│   └── fewshot_examples.json
├── results/                             # output evaluasi per model (report dilacak; png/jsonl di-ignore)
├── eval.py                              # evaluasi benchmark
├── generate_reasoning.py                # distilasi reasoning train (guru)
├── generate_reasoning_validation.py     # distilasi reasoning val (guru)
├── build_sft.py                         # bangun dataset SFT dari CSV + reasoning
├── merge_trainval.py                    # gabung train+val (val reindex 960–1079)
├── requirements.txt                     # dependensi langsung (versi longgar >=)
└── requirements.lock.txt                # full pin (==) utk reproducibility
```

---

## Setup

```bash
cd ~/Yusuf/Kuliah/ProjectSemester7/Cognitive-Classification-Using-LLM

python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

---

## Menjalankan llama-server

Server binary ada di `~/llama.cpp/build/bin/llama-server`. Setup yang direkomendasikan (Qwen 35B):

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/Yusuf/Model/Qwen/Qwen3.6-35B-A3B-apex-iquality.gguf \
  -c 32768 --host 127.0.0.1 --port 8889 \
  -ngl 99 -t 6 -b 4096 -ub 2048 -fa on \
  --n-cpu-moe 39 \
  --cache-type-k q4_0 --cache-type-v q8_0 \
  --load-mode mlock --cache-ram 4096 --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20
```

**Catatan penting:**
- Gunakan `-c 32768` (bukan 262144) untuk evaluasi — KV cache besar bikin lambat & risiko OOM.
- Jangan pakai `--ctx-checkpoints` — bikin prefill O(n²) patologis.
- Untuk model kecil (Nanbeige 3B, Qwythos 9B): pertimbangkan `--max-tokens 12000` di eval.py karena thinking panjang.

---

## Menjalankan evaluasi

```bash
cd ~/Yusuf/Kuliah/ProjectSemester7/Cognitive-Classification-Using-LLM

# Few-shot (definisi + contoh + aturan — akurasi tertinggi)
venv/bin/python3 eval.py --prompt few --fresh \
  --out "results/Qwen3.6-35B-A3B/Pre-Finetune/Few-Shot"

# Zero-shot (definisi kelas saja — uji pemahaman murni)
venv/bin/python3 eval.py --prompt zero --fresh \
  --out "results/Qwen3.6-35B-A3B/Pre-Finetune/Zero-Shot"

# Uji cepat 5 sampel
venv/bin/python3 eval.py --prompt few --limit 5 \
  --out "/tmp/test"

# Resume dari checkpoint (skip --fresh)
venv/bin/python3 eval.py --prompt zero \
  --out "results/Qwen3.6-35B-A3B/Pre-Finetune/Zero-Shot"

# Lihat prompt tanpa hubungi server
venv/bin/python3 eval.py --prompt few --dry-run

# Budget token lebih besar (untuk model "overthinker")
venv/bin/python3 eval.py --prompt few --max-tokens 12000 --max-budget 16000 \
  --out "results/Qwythos/Pre-Finetune/Few-Shot"
```

---

## Fitur Utama `eval.py`

### Auto-escalate budget token
Kalau generation kepotong di `max_tokens` (`finish_reason=length`), budget otomatis digandakan sampai `--max-budget`. Default: 4096 → 8192.

### Checkpoint / Resume
Progress disimpan per-sampel di `_ckpt_{variant}.jsonl`. Gunakan **tanpa `--fresh`** untuk melanjutkan dari sampel terakhir yang selesai. Kalau server mati di tengah jalan, tinggal restart server lalu jalankan ulang command yang sama.

### 3-lapis parser output JSON
1. `json.loads` (JSON murni) → 2. regex `"label"` + `"alasan"` → 3. regex `C[0-4]`

### Response format auto-detection
`probe_format()` mencoba `json_schema` → `json_object` → `none` (plain text), pilih yang menghasilkan parse-able output.


---

## Output (`results/.../`)

| File | Isi |
|---|---|
| `report_{variant}.txt` | accuracy, macro/weighted-F1, per-class precision/recall/F1, waktu total |
| `predictions_{variant}.jsonl` | per-sampel: idx, text[:500], gold, pred, alasan, thinking_head[:500], latency, finish_reason |
| `confusion_matrix_{variant}.png` | heatmap 5×5 (prediksi vs label emas) |

`{variant}` = `fewshot` atau `zeroshot`.

---

## Catatan Metodologis

- **Prompt decontaminated**: teks test-set tidak ada di prompt (mini-examples di rules diparafrase). Model belajar pola + alasan, bukan hafal teks.
- **Class imbalance**: C4 cuma 5 sampel — Macro-F1 lebih meaningful dari akurasi mentah.
- **Parse failure**: dilaporkan `PARSE_FAIL` dan dihitung dalam `acc_strict` (parse-fail = salah).
- **Seed 42 deterministik**: `seed=42` di setiap request, sampling tidak di-override dari server.

---

## Persiapan Fine-tuning

**Distilasi reasoning** (teacher = Qwen3.6-35B-A3B, Mode Guru). Guru menulis alasan klasifikasi konsisten dengan label emas mengikuti rubrik `prompts/guru_system.txt`:
- `generate_reasoning.py` / `generate_reasoning_validation.py` → reasoning train (960) + val (120)
- `merge_trainval.py` → gabungkan (val direindex 960–1079) → 1080 sampel
- `build_sft.py` → `data/dataset_sft_sharegpt.jsonl` (format ShareGPT, siap fine-tune)

Distribusi label gabungan: **C0=246 · C1=197 · C2=168 · C3=413 · C4=56**.

**Perbaikan label emas (25 entri).** Sebelum distilasi, label diaudit ulang vs rubrik:
- Train (12): 2 bug sinkronisasi reasoning↔gold, 5 proof-of-axiom C2→C3, 1 C4→C3, 4 borderline.
- Val (13): 10 C3→C2 (rencana/asersi/daftar over-C3), 1 C3→C1, 2 C0→C2 (koreksi konseptual).
- Catatan: bias C2/C3 **berlawanan** antar split (train under-C3, val over-C3).

Rincian per-entry: `NOTE.md` (lokal, tidak dipush).
