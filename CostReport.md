# CostReport — API DeepSeek vs Sewa GPU untuk Forum Diskusi Ber-AI

**Studi biaya-kecepatan-akurasi** untuk deployment AI moderator forum diskusi kelompok (1 mata kuliah).
Sumber: pengukuran sendiri (benchmark 3 GPU RunPod + log llama-server + token real via API) dan harga resmi
`api-docs.deepseek.com` (Nov 2026). Semua angka biaya dalam USD.

---

## 1. Kesimpulan Eksekutif

| Skenario | Biaya/semester | Akurasi (5-kelas CoI) | Speed @40 user | Ops |
|---|---|---|---|---|
| **DeepSeek API V4-Flash** | **~$7–15** | **0.9500** | 100+ t/s/user, concurrency 2.500 | nol |
| GPU A40 (1 jam/hari) | $43 | ≤0.9500 (model lebih kecil) | 1–2 t/s/user ✗ | cold load + patch |
| GPU RTX PRO 6000 | $205 | idem | ~3 t/s/user ✗ | idem |
| GPU H100 NVL | $313 | idem | ~7 t/s/user ✗ | idem |
| GPU multi-card (5× H100 NVL) | **~$1.570** | idem | 35 t/s/user ✓ | berat |

**API menang di semua sumbu yang diukur.** GPU lokal hanya relevan bila privasi menjadi
constraint keras kampus (mitigasi: alias + name-scrubbing — lihat §8).

---

## 2. Beban Kerja (asumsi Pak Arik)

- Batas atas **140 topik diskusi/semester**; per topik: 10 mahasiswa + dosen + asdos + 1 AI.
- Perkiraan post mahasiswa: 140 × 10 × 2–3 ≈ **2.800–4.200 post/semester** (dipakai 4.000).
- Dua agent:
  - **Klasifikator** — tiap post mhs → label C0–C4 + alasan (background, tanpa SLO).
  - **Moderator** — baca transcript thread, putuskan intervensi (schedule harian 19.00 / immediate).
- Mode lokal: **1 jam/hari** → 98 jam/semester (7 hari/mgg) atau 70 jam (5 hari/mgg).
- Kebutuhan riil komputasi harian: ~15–20 menit (43 post/hari ÷ ~8 slot) — 1 jam/hari = overkill 3×.

## 3. Beban Token (terukur, bukan estimasi)

| Komponen | Token | Catatan |
|---|---|---|
| System prompt klasifikator (few-shot) | **7.751** | identik semua request → cache-hit |
| 1 request klasifikasi (system + post) | ~8.000 in / ~2.000 out | output didominasi thinking (default `high`) |
| System prompt moderator | **924** | `prompts/moderator_system.md` |
| 1 run moderator (thread 10 post real) | **2.622 in / 2.877 out** | diukur via `usage` API |

## 4. Harga API DeepSeek (resmi, per 1M token)

| Komponen | Peak | Off-peak (½) |
|---|---|---|
| Input cache-hit | $0.014 | $0.007 |
| Input cache-miss | $0.44 | $0.22 |
| Output (termasuk thinking) | $1.32 | $0.66 |

Peak = 01:00–04:00 & 06:00–10:00 UTC → **batch jam 19.00 WIB otomatis off-peak**.
Concurrency limit: 2.500.

### Biaya per unit (peak, dengan context-cache aktif)

| Operasi | Hitungan | $/unit |
|---|---|---|
| 1 klasifikasi (8K in / 2K out) | 7.75K cache + 0.3K miss + 2K out | **$0.0029** |
| 1 run moderator | 0.9K cache + 1.7K miss + 2.9K out | **$0.0046** |

### Total per semester (1 matkul)

| Komponen | Peak | Off-peak |
|---|---|---|
| Klasifikasi 4.000 post | $11.50 | $5.80 |
| Moderator 560 run (140 topik × 4) | $2.60 | $1.30 |
| **Total** | **$14.10** | **$7.10** |

Sensitivitas: volume 3× lipat (12.000 post) → masih $21–42/semester.
Lever tambahan bila akurasi memungkinkan: `thinking=off` → biaya output turun ~5× (total → ~$4),
dan zero-shot (prompt pendek) — terukur: few-shot 0.9500 vs zero-shot 0.6833 (default) / 0.7333 (thinking max).

> Catatan kanal ketiga: subscription OpenCode (Zen) = $0 marginal per request, namun tanpa SLA
> (latensi teramati 6–40 dtk, variance antrean) — tidak direkomendasikan untuk produksi.

## 5. Sewa GPU (RunPod, harga aktual)

| GPU | VRAM | $/jam | Semester (98j / 70j) |
|---|---|---|---|
| A40 | 48 GB | $0.44 | $43.12 / $30.80 |
| RTX PRO 6000 Blackwell | 96 GB | $2.09 | $204.82 / $146.30 |
| H100 NVL | 94 GB | $3.19 | $312.62 / $223.30 |

### Hasil benchmark sendiri (llama-server, 5 konkuren, MTP draft-3, KV q8_0)

| Model | Decode t/s per stream (A40 / PRO 6000 / H100 NVL) | VRAM peak |
|---|---|---|
| Qwen3.5-9B Q8 | 40.9 / 72.0 / 125.7 | 18.7 GB |
| Qwen3.8-27B-Fable Q4 | 21.5 ¹ / 32.7 / 45.0 | 34.5 GB |
| Ornith1.5-35B-A3B Q4 | 40.0 ¹ / 69.8 ¹ / 86.6 | 28.3 GB |

¹ run A40 memakai varian same-base (GAIN / BigBang); kelas ukuran identik.
Detail: `eksperimen/BENCHMARK.md`. Temuan: workload **prefill-bound** (prompt 3K–23K tok >> output);
MTP acceptance 0.68–0.81 kecuali Ornith@H100 (0.36 — nyaris sia-sia).

## 6. Concurrency & SLO — titik kematian GPU

SLO minimum yang ditetapkan: **35 t/s per user**. Kelas 100 mhs, mode immediate → burst **40 request konkuren**.

Decode = bandwidth-bound → aggregate ceiling ≈ BW ÷ bobot aktif; per-stream ≈ ceiling ÷ N:

| Platform | Aggregate real | @40 stream | Lolos SLO? |
|---|---|---|---|
| A40 + 9B | ~87 t/s | **2.2 t/s** | ✗ |
| PRO 6000 + 9B | ~132 t/s | 3.3 t/s | ✗ |
| H100 NVL + 9B | ~290 t/s | 7.3 t/s | ✗ (lolos hanya s.d. ~8 stream) |
| **DeepSeek API** | cluster | **100+ t/s** | ✓ (limit 2.500) |

Memenuhi SLO 40-user secara lokal ≈ 5× H100 NVL ≈ $16/jam ≈ **$1.570/semester** (1 jam/hari).
Mengingat API dibayar per **token** (bukan per slot), concurrency tidak memengaruhi biaya sama sekali.

## 7. Akurasi (benchmark 120 sampel, `eval.py`, seed 42)

| Model | Akurasi | Macro-F1 |
|---|---|---|
| **DeepSeek-V4-Flash (API, thinking high)** | **0.9500** | 0.9172 |
| DeepSeek-V4-Flash (thinking max) | 0.9417 | 0.9080 — overthinking, 5× latensi |
| BigBang 35B-A3B (lokal) | 0.9500 | 0.9594 |
| Qwen3.8-27B-Fable (lokal) | 0.9500 | 0.9225 |
| Ornith1.5-35B-A3B (lokal) | 0.9250 | 0.9066 |
| Kandidat lokal murah (9B class) | 0.8833 | 0.8288 |

API memberi akurasi puncak dengan model terbesar; `thinking=max` kontraproduktif di few-shot
(0.9500→0.9417) namun membantu di zero-shot (0.6833→0.7333) — reasoning panjang hanya
berharga saat prompt minim. Kandidat lokal murah (9B class) 0.8833 — selisih 6.7 pp dari puncak.

## 8. Kapan GPU lokal masih masuk akal

1. **Privasi** — kampus melarang isi diskusi keluar (mitigasi tersedia: pseudonim alias +
   scrubbing nama dari roster → risiko turun drastis, API tetap layak).
2. **Mode batch murni tanpa SLO** (klasifikasi malam + moderator terjadwal) — di sini A40
   $43/semester valid dan kompetitif, tetap 3–6× API.
3. Hardware sudah dimiliki (biaya marginal listrik saja).
4. Eksperimen/riset gratis via kredit provider (mis. Modal $30/bln) — bukan skema produksi.

## 9. Rekomendasi

**Gunakan DeepSeek API direct (deepseek-v4-flash, thinking default) untuk produksi 1 matkul.**
Biaya ~$7–15/semester, akurasi tertinggi (0.9500), speed konsisten 100+ t/s pada concurrency berapa pun,
tanpa ops. Jalankan batch klasifikasi + moderator pada jam off-peak (19.00 WIB) untuk harga setengah.
Sediakan fallback A40/Modal hanya bila kebijakan privasi berubah.

---

*Data mentah: `eksperimen/{A40,H100 NVL,RTX PRO 6000}/RINGKASAN.md`, `eksperimen/BENCHMARK.md`,
`results/DeepSeek-V4-Flash/*/report_*.txt`, prompt `prompts/moderator_system.md`.
Harga RunPod & DeepSeek diverifikasi Nov 2026 — cek ulang sebelum digunakan di laporan final.*
