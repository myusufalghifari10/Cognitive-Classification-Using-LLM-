import csv, json, os

CSV_PATH = "data/trainval_dataset.csv"
REASONING_PATH = "results/teacher_reasoning.jsonl"
TEST_PROMPT_PATH = "prompts/test_system.txt"

os.makedirs("results", exist_ok=True)
OUT_SFT = "results/dataset_sft_sharegpt.jsonl"

with open(TEST_PROMPT_PATH, "r", encoding="utf-8") as f:
    TEST_SYSTEM = f.read()

reasoning_map = {}
if os.path.exists(REASONING_PATH):
    with open(REASONING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                reasoning_map[data["idx"]] = data["reasoning"]
            except:
                pass

with open(CSV_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

count = 0
with open(OUT_SFT, "w", encoding="utf-8") as f:
    for i, row in enumerate(rows):
        idx = int(row.get("id", i))
        if idx not in reasoning_map: continue
        
        # Format output assistant HARUS sama dengan FORMAT JAWABAN di prompt testing
        assistant_content = json.dumps({
            "alasan": reasoning_map[idx], 
            "label": row["label"]
        }, ensure_ascii=False)
        
        sft_sample = {
            "messages": [
                {"role": "system", "content": TEST_SYSTEM},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
        count += 1

print(f"🎉 Dataset SFT berhasil dibuat di {OUT_SFT} ({count} sampel)")
