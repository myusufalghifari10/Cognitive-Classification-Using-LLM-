import csv, json, os, shutil, collections

# Sumber (jangan diubah oleh script ini)
TRAIN_CSV = "data/training_dataset.csv"            # 960 baris, idx 0-959
VAL_CSV   = "data/validation_dataset.csv"          # 120 baris, idx 0-119
TRAIN_RSN = "results/teacher_reasoning.jsonl"      # 960 (idx 0-959) -> target merge
VAL_RSN   = "results/teacher_reasoning_validation.jsonl"  # 120 (idx 0-119) -> sumber val

# Output
OUT_CSV = "data/trainval_dataset.csv"              # CSV gabungan agar build_sft bisa join idx 960-1079

os.makedirs("results", exist_ok=True)

# --- 1. Baca CSV; OFFSET = jumlah baris train (val idx 0 -> OFFSET) ---
with open(TRAIN_CSV, encoding="utf-8") as f:
    _r = csv.DictReader(f); fields = _r.fieldnames; train_rows = list(_r)
with open(VAL_CSV, encoding="utf-8") as f:
    val_rows = list(csv.DictReader(f))
OFFSET = len(train_rows)   # 960

# --- 2. Baca reasoning: train (idx<OFFSET, supaya idempoten) + val ---
with open(TRAIN_RSN, encoding="utf-8") as f:
    train_rsn = [e for e in (json.loads(l) for l in f if l.strip()) if e["idx"] < OFFSET]
with open(VAL_RSN, encoding="utf-8") as f:
    val_rsn = [json.loads(l) for l in f if l.strip()]

# --- 3. Merge reasoning: train apa adanya + val direindex ke OFFSET.. ---
merged_rsn = list(train_rsn)
for e in sorted(val_rsn, key=lambda x: x["idx"]):
    merged_rsn.append({"idx": e["idx"] + OFFSET, "label": e["label"], "reasoning": e["reasoning"]})
merged_rsn.sort(key=lambda e: e["idx"])

# --- 4. Merge CSV: baris train + baris val (idx = urutan baris) ---
merged_rows = train_rows + val_rows

# --- 5. Backup premerge (sekali, jangan timpa) ---
pre = TRAIN_RSN + ".premerge"
if not os.path.exists(pre):
    shutil.copy2(TRAIN_RSN, pre)

# --- 6. Tulis ---
with open(TRAIN_RSN, "w", encoding="utf-8") as f:
    for e in merged_rsn:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(merged_rows)

# --- 7. Verifikasi: idx unik, join lengkap, label konsisten ---
rsn_by = {e["idx"]: e for e in merged_rsn}
assert len(rsn_by) == len(merged_rsn), "idx duplikat di reasoning!"
bad = [(i, r["label"], rsn_by.get(i, {}).get("label"))
       for i, r in enumerate(merged_rows)
       if i not in rsn_by or rsn_by[i]["label"] != r["label"]]

print(f"OFFSET (n train)  : {OFFSET}")
print(f"reasoning merged  : {len(merged_rsn)}  (train {len(train_rsn)} + val {len(val_rsn)})  -> {TRAIN_RSN}")
print(f"CSV merged        : {len(merged_rows)}  -> {OUT_CSV}")
print(f"idx range         : 0..{len(merged_rows)-1}  (val kini {OFFSET}..{OFFSET+len(val_rows)-1})")
print(f"masalah join/label: {len(bad)}  (harus 0)")
print(f"distribusi label  : {dict(sorted(collections.Counter(r['label'] for r in merged_rows).items()))}")
print(f"\nPremerge backup   : {pre}  (train reasoning murni {OFFSET} baris)")
