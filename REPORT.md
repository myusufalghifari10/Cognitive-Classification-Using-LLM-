# Cognitive Presence Classification — Few-Shot vs Zero-Shot Evaluation Report

Sembilan model LLM diuji pada 120 postingan forum diskusi mahasiswa menggunakan prompt few-shot (`prompts/system_few_shot.md`) dan zero-shot (`prompts/system_zero_shot.md`). Seluruh evaluasi menggunakan `seed=42` dan sampling default server.

---

## Tabel 1 — Few-Shot (definisi + contoh + aturan)

| Model | Size | Acc | Prec (M) | Rec (M) | F1 (M) | F1 (W) | t/sampel |
|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | 35B A3B | **0.9417** | 0.9523 | 0.9109 | 0.9290 | 0.9417 | 45.4s |
| **KAT-Coder2.5** | 35B A3B | 0.9250 | 0.9349 | 0.9326 | **0.9332** | 0.9262 | 21.4s |
| **Qwen3.6-35B-A3B-Fable** | 35B A3B | 0.9250 | 0.9131 | 0.9206 | 0.9063 | 0.9233 | 65.9s |
| **Gemma4-26B-A4B-IT** | 26B A4B | 0.9000 | 0.8387 | 0.8870 | 0.8557 | 0.9036 | 54.0s |
| **Qwen3.6-35B-A3B-DSV4Pro** | 35B A3B | 0.8750 | 0.8270 | 0.8995 | 0.8434 | 0.8804 | 33.2s |
| **Qwen3.5-9B-Fable** | 9B | 0.8833 | 0.9119 | 0.8089 | 0.8288 | 0.8843 | 36.7s |
| **Nanbeige4.2** | 4B | 0.8000 | 0.7543 | 0.7875 | 0.7577 | 0.8108 | 62.6s |
| **Qwythos** | 9B | 0.7667 | 0.8070 | 0.7211 | 0.7458 | 0.7896 | 68.2s |
| **GLM-4.7-Flash** | 30B A3B | 0.7750 | 0.8668 | 0.7134 | 0.7154 | 0.7870 | 91.4s |

> Prec/Rec = Macro avg; F1(M) = Macro-F1; F1(W) = Weighted-F1. Accuracy & strict accuracy identik untuk semua model (0 `parse_fail`) kecuali Qwythos (3 `parse_fail`, strict=0.7667).

---

## Tabel 2 — Zero-Shot (hanya definisi kelas, tanpa contoh & aturan)

| Model | Size | Acc | Prec (M) | Rec (M) | F1 (M) | F1 (W) | t/sampel |
|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B-Fable** | 35B A3B | **0.7750** | 0.7005 | 0.7253 | **0.7094** | 0.7784 | 25.6s |
| **Qwen3.6-35B-A3B** | 35B A3B | 0.7667 | 0.7212 | 0.7014 | 0.7062 | 0.7684 | 41.3s |
| **Qwen3.5-9B-Fable** | 9B | 0.7583 | 0.6885 | 0.6819 | 0.6794 | 0.7638 | 37.5s |
| **KAT-Coder2.5** | 35B A3B | 0.7417 | 0.6679 | 0.7189 | 0.6788 | 0.7521 | 10.0s |

---

## Temuan Singkat

### 1. Few-shot: Qwen akurasi tertinggi, KAT paling efisien
- **Qwen3.6-35B-A3B** akurasi tertinggi (0.9417), Macro-F1 0.9290. Ketat di semua kelas, C4 precision 1.0.
- **KAT-Coder2.5** Macro-F1 **tertinggi** (0.9332) dengan waktu hanya **21.4s/sampel** — 2× lebih cepat dari Qwen, C4 F1 sempurna (1.0). Model paling efisien di benchmark ini.
- Tren skala terlihat jelas: 4B (Nanbeige, 0.80) → 9B (Qwythos, 0.77) → 26B+ (0.88-0.94). GLM-4.7-Flash terlemah di kelas besar (0.775) dengan C2 precision 0.44 — over-prediksi C2.

### 2. Zero-shot: semua jatuh ~15-20 poin, Fable paling tangguh
- Zero-shot menghilangkan aturan + contoh → akurasi turun drastis: Qwen 0.942 → 0.767 (−0.175), KAT 0.925 → 0.742 (−0.183), Fable 0.925 → 0.775 (−0.150).
- **Fable paling tangguh di zero-shot** (0.775), mengungguli Qwen asli (0.767) dan KAT (0.742) — distilasi mempertahankan pemahaman definisi kelas lebih baik.
- **Qwen3.5-9B-Fable** (9B) juga tangguh di zero-shot: 0.7583, hanya −0.125 dari few-shot (0.8833) — lebih kecil dari Qwen3.6-35B-A3B (−0.175) dan KAT (−0.183). Tapi C4 Resolusi F1 jebol di zero-shot (0.222 vs 0.571 di few-shot) karena hanya 5 sampel + tidak ada aturan PEMBEDA.
- Semua model kesulitan di C2/C3 boundary tanpa aturan PEMBEDA — Eksplorasi F1 0.53-0.58 (vs 0.82-0.88 di few-shot).

### 3. Catatan teknis
- Seluruh evaluasi menggunakan prompt decontaminated (tidak ada teks test-set di prompt). Angka dapat direproduksi dengan `seed=42`.
- Beberapa model memerlukan penyesuaian konteks & budget token (Nanbeige & Qwythos: thinking sangat panjang; KAT & Gemma: `json_schema`; Qwen apex: `none`). Qwen3.5-9B-Fable menggunakan `json_schema` dengan budget 8192.
- KAT-Coder2.5 zero-shot memiliki 1 `parse_fail` (timeout).

---

*Prompt: `prompts/system_few_shot.md` & `prompts/system_zero_shot.md`.*
