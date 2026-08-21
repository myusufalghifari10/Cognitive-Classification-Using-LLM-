# ProgressReport — Cognitive Classification Using LLM

**Status per 21 Agustus 2026.** Pipeline end-to-end: benchmark 20 run pre-finetune →
perbaikan data → distill reasoning guru → fine-tune → eval ulang → studi cost/speed deployment.
Basis kode: `eval.py` (evaluasi), `train.py` (fine-tune), `generate_reasoning*.py` (distilasi),
`pod/` (RunPod automation). Test set: 120 postingan, label emas anotator, tidak pernah diubah.

---

## 1. Dari Mana Data

- **Sumber**: Data dari Pak Arik.
- **Split**: TRAIN 960 · VALIDATION 120 · TEST 120 (test = label emas anotator manusia, tidak tersentuh).
- **Distribusi test**: C0:29 · C1:18 · C2:20 · C3:48 · C4:5 (C4 langka → Macro-F1 metrik utama).

## 2. Benchmark Pre-Finetune — 22 run, 17 model unik

Semua via `eval.py` (llama-server lokal / API OpenCode Zen untuk DeepSeek), seed 42,
prompt few-shot & zero-shot decontaminated, output JSON tervalidasi, PARSE_FAIL = salah.

### Few-Shot (lengkap, 20 run)

| Model | Acc | Macro-F1 | W-F1 |
|---|---|---|---|
| BigBang 35B-A3B | **0.9500** | 0.9594 | 0.9520 |
| Qwen3.8-27B-Fable | **0.9500** | 0.9225 | 0.9500 |
| **DeepSeek-V4-Flash (API)** | **0.9500** | 0.9172 | 0.9513 |
| Qwen3.6-35B-A3B | 0.9417 | 0.9290 | 0.9417 |
| DeepSeek-V4-Flash max-think | 0.9417 | 0.9080 | 0.9427 |
| Qwen3.6-27B-Fable | 0.9333 | 0.9085 | 0.9336 |
| KAT-Coder2.5 | 0.9250 | 0.9332 | 0.9262 |
| Ornith1.5-35B-A3B | 0.9250 | 0.9066 | 0.9265 |
| Qwen3.6-35B-A3B-Fable | 0.9250 | 0.9063 | 0.9233 |
| Qwen3.8-27B-GAIN | 0.9083 | 0.9029 | 0.9084 |
| Qwen3.8-27B base | 0.9083 | 0.8801 | 0.9103 |
| Gemma4-26B-A4B-IT | 0.9000 | 0.8557 | 0.9036 |
| Qwen3.6-40B | 0.8917 | 0.8604 | 0.8905 |
| Qwen3.5-9B-Fable | 0.8833 | 0.8288 | 0.8843 |
| Qwen3.6-35B-A3B-DSV4Pro | 0.8750 | 0.8434 | 0.8804 |
| Muse-Glimmer | 0.8583 | 0.8294 | 0.8625 |
| Nanbeige4.2-3B | 0.8000 | 0.7577 | 0.8108 |
| Nemotron | 0.7881 | 0.7026 | 0.7858 |
| Qwythos | 0.7863 | 0.7458 | 0.7896 |
| GLM-4.7-Flash | 0.7750 | 0.7154 | 0.7870 |

*(Zero-shot & detail per kelas: `README.md` / `results/*/report_*.txt`.)*

### Zero-Shot (5 model teratas; lengkap di README)

| Model | Acc | Macro-F1 |
|---|---|---|
| Qwen3.6-35B-A3B-Fable | **0.7750** | 0.7094 |
| Qwen3.6-35B-A3B | 0.7667 | 0.7062 |
| Qwen3.5-9B-Fable | 0.7583 | 0.6794 |
| KAT-Coder2.5 | 0.7479 | 0.6788 |
| BigBang | 0.7059 | 0.6549 |
| DeepSeek-V4-Flash max-think | 0.7333 | 0.6396 |
| DeepSeek-V4-Flash (default) | 0.6833 | 0.6072 |

Temuan: (a) tiga model share puncak few-shot 0.9500 — MoE 35B lokal, dense 27B distilled, API cloud;
(b) `thinking=max` pada DeepSeek: few-shot **turun** (0.9500→0.9417, overthinking) tapi zero-shot
**naik** (0.6833→0.7333) — reasoning panjang hanya membantu saat prompt minim;
(c) rentang few-shot 0.775–0.950 menunjukkan task sensitif terhadap model;
(d) tanpa contoh, DeepSeek (0.7333) kalah dari MoE lokal 0.7750 — kelas C4 recall rendah (0.20) di semua varian.

## 3. Perbaikan Data (9 Agu — `NOTE.md`)

Audit independen 960 TRAIN + 120 VAL terhadap rubrik `guru_system.txt`:
- **TRAIN: 12 label diperbaiki** (2 mismatch guru-vs-gold beracun, 5 C2→C3 proof-of-axiom,
  1 C4→C3 mid-discussion, 4 borderline) → 0 mismatch, 0 reasoning terkontaminasi.
- **VAL: 13 label diperbaiki** (10 C3→C2 asersi/rencana tanpa derivasi, 1 C3→C1, 2 C0→C2).
- **TEST tidak disentuh**; semua perubahan berbackup `.bak` + revert command di NOTE.
- **Temuan sistematis**: bias C2/C3 berlawanan antar split (train under-C3, val over-C3)
  → risiko sinyal kontradiktif saat merge; dicatat sebagai risiko mitigasi lanjutan.

## 4. Distilasi Reasoning (Mode Guru)

`generate_reasoning.py` (+ versi validation): teacher Qwen3.6-35B-A3B-Fable menulis alasan klasifikasi
konsisten dengan label emas mengikuti rubrik → `teacher_reasoning*.jsonl` → `build_sft.py`
→ `dataset_sft_sharegpt.jsonl` (**1080 sampel** = 960 train + 120 val post-audit,
format ShareGPT messages) → `merge_trainval.py` menjaga idempotensi idx.

## 5. Fine-tune (Unsloth, pod RunPod)

`train.py` di pod `pod/00–03_*.sh` (RUNBOOK lengkap di `pod/RUNBOOK.md`):
- **Qwen3.5-9B-Defiant** — LoRA 16-bit (QLoRA di-avoid per rekomendasi Unsloth utk Qwen3.5),
  3 epoch, lr 2e-4, max-seq 12288 → adapter + merge 16-bit + **GGUF Q8_0** otomatis.
- **Qwen3.6-35B-A3B** — QLoRA 4-bit (72GB BF16 tidak muat; berkas pod `02_finetune.sh`).
- Perjalanan debugging tercatat di git history (OOM → expandable_segments + paged optimizer;
  319s/it → normal; DataCollator API trl 0.24; pinning cu128 utk Blackwell).

## 6. Hasil Post-Finetune

| Model | Metode | Few-Shot | Zero-Shot |
|---|---|---|---|
| Qwen3.5-9B-Fable | LoRA 16-bit | **0.9250** / F1 0.9165 | 0.7917 / F1 0.7596 |
| Qwen3.6-35B-A3B | QLoRA 4-bit | **0.9250** / F1 0.9123 | 0.7917 / F1 0.7722 |

- 9B naik **+4.2 poin** (0.8833 → 0.9250) dengan budget 9B — distilasi rubrik efektif.
- 35B turun tipis di few-shot (0.9417 → 0.9250) tapi zero-shot naik +2.5 poin (0.7667 → 0.7917)
  → fine-tune paling bernilai saat prompt pendek.
- Baseline kompetitif (BigBang/27B-Fable/DSV4 API 0.9500) belum terlampaui — model hasil
  fine-tune mengejar kelas atas dengan biaya inferensi jauh lebih murah.

## 7. Studi Cost & Speed Deployment (eksperimen RunPod)

Pertanyaan: API DeepSeek vs sewa GPU vs subscription, untuk beban 1 matkul
(140 topik, ±4.000 post/semester, dua agent: klasifikator + moderator).

1. **Benchmark speed** (3 GPU × 3 model, 5 konkuren, MTP, KV q8_0 — `eksperimen/BENCHMARK.md`):
   A40 41–72 t/s per stream kelas 9B; H100 NVL tercepat 126 t/s; workload prefill-bound.
2. **SLO 35 t/s @ 40 user**: semua single-GPU gagal (1–7 t/s); lolos perlu ±5× H100 ≈ $1.570/smt.
3. **API DeepSeek**: ~$7–15/semester (token terukur: klasifikasi $0.0029/post,
   moderator $0.0046/run; prompt terukur 7.751/924 token), concurrency 2.500, 100+ t/s konsisten.
4. **Kesimpulan (`CostReport.md`)**: API menang di semua sumbu; GPU lokal hanya jika privasi
   jadi constraint keras (mitigasi alias + scrubbing tersedia); A40 = fallback termurah $43/smt.

## 8. Dua Agent Produksi (desain awal)

- **Klasifikator**: arsitektur = eval pipeline (system few-shot + 1 post → JSON label+alasan).
- **Moderator**: `prompts/moderator_system.md` (924 tok) — 6 aturan intervensi CoI
  (stagnan/pertanyaan terbuang/menanya ke AI/hampir selesai/off-topic/sepi), output JSON,
  gaya sokratik, dilarang menjawab materi. Sudah diuji 1 run real via API: deteksi benar,
  target tepat, JSON valid. **Kontaminasi konteks antar-request terverifikasi nol** (API stateless,
  `build_messages()` selalu fresh `[system, user]`).

## 9. Yang Sedang Berjalan / Berikutnya

- [ ] Integrasi moderator ke forum (web app) — belum dimulai.


---

*Semua klaim dapat ditelusuri: report di `results/*/`, log di `eksperimen/*/RINGKASAN.md`,
perubahan data di `NOTE.md` + `.bak`, riwayat di `git log`.*
