# Cognitive Classification Using LLM

Benchmark 9 model LLM (Qwen, KAT-Coder, Gemma, GLM, Nanbeige, Qwythos, Fable) untuk klasifikasi **Cognitive Presence** 5-kelas pada 120 postingan forum diskusi mahasiswa. Model dijalankan via **llama.cpp (`llama-server`)** dengan prompt decontaminated (few-shot & zero-shot).

Repo ini melanjutkan riset `cognitive-classification` (IndoBERT) dengan **LLM generik sebagai baseline**, dan menjadi fondasi untuk *Prescriptive Learning Analytics dengan Agentic AI*.

---

## Benchmark Results

### Few-Shot (definisi + contoh + aturan, `system_few_shot.md`)

| Model | Size | Acc | Macro-F1 | Weighted-F1 | t/sampel |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | 35B A3B | **0.9417** | **0.9290** | 0.9417 | 45.4s |
| **KAT-Coder2.5** | 35B A3B | 0.9250 | **0.9332** | 0.9262 | 21.4s |
| **Qwen3.6-35B-A3B-Fable** | 35B A3B | 0.9250 | 0.9063 | 0.9233 | 65.9s |
| **Gemma4-26B-A4B-IT** | 26B A4B | 0.9000 | 0.8557 | 0.9036 | 54.0s |
| **Qwen3.6-35B-A3B-DSV4Pro** | 35B A3B | 0.8750 | 0.8434 | 0.8804 | 33.2s |
| **Qwen3.5-9B-Fable** | 9B | 0.8833 | 0.8288 | 0.8843 | 36.7s |
| **Nanbeige4.2** | 4B | 0.8000 | 0.7577 | 0.8108 | 62.6s |
| **Qwythos** | 9B | 0.7667 | 0.7458 | 0.7896 | 68.2s |
| **GLM-4.7-Flash** | 30B A3B | 0.7750 | 0.7154 | 0.7870 | 91.4s |

### Zero-Shot (hanya definisi kelas, `system_zero_shot.md`)

| Model | Size | Acc | Macro-F1 | Weighted-F1 | t/sampel |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B-Fable** | 35B A3B | **0.7750** | **0.7094** | 0.7784 | 25.6s |
| **Qwen3.6-35B-A3B** | 35B A3B | 0.7667 | 0.7062 | 0.7684 | 41.3s |
| **Qwen3.5-9B-Fable** | 9B | 0.7583 | 0.6794 | 0.7638 | 37.5s |
| **KAT-Coder2.5** | 35B A3B | 0.7417 | 0.6788 | 0.7521 | 10.0s |

> Lihat `REPORT.md` untuk analisis lengkap. Semua run pakai `seed=42`, prompt v8 decontaminated (tidak ada teks test-set di prompt).

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
│   └── test_dataset.csv           # 120 sampel test set
├── prompts/
│   ├── system_few_shot.md         # prompt few-shot (definisi + aturan + 13 contoh)
│   ├── system_zero_shot.md        # prompt zero-shot (hanya definisi kelas)
│   ├── user.md                    # template user prompt ({text} placeholder)
│   └── fewshot_examples.json      # sumber 13 contoh few-shot (dari train set)
├── results/                       # output evaluasi per model per prompt
├── eval.py                        # script evaluasi utama
├── requirements.txt               # dependensi Python
└── REPORT.md                      # ringkasan benchmark 9 model
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
