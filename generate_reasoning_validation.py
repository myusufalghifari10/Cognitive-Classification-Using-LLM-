import csv, json, re, time, requests, os

API_URL   = "http://127.0.0.1:8889/v1/chat/completions"
CSV_PATH  = "data/validation_dataset.csv"
PROMPT_PATH = "prompts/guru_system.txt"

# Paksa buat folder results agar tidak nyasar ke root
os.makedirs("results", exist_ok=True)
# OUT_PATH DIPISAH dari train (results/teacher_reasoning.jsonl) agar tidak menimpa
OUT_PATH  = "results/teacher_reasoning_validation.jsonl"

BATCH     = 10
MAX_RETRY = 3

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open(CSV_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Load checkpoint
done_ids = set()
if os.path.exists(OUT_PATH):
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["idx"])
            except:
                pass

def extract_json_array(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        return json.loads(text[start:end+1])
    raise ValueError("JSON Array tidak ditemukan")

print(f"[VALIDATION] Memulai proses. Total sampel: {len(rows)}. Sudah selesai: {len(done_ids)}")

for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    samples = []
    expected_golds = {}

    for j, r in enumerate(batch):
        idx = int(r.get("id", i+j))
        samples.append({"idx": idx, "text": r["text"], "gold": r["label"]})
        expected_golds[idx] = r["label"]

    if all(s["idx"] in done_ids for s in samples):
        continue

    user_msg = f"Buatkan reasoning untuk {len(samples)} sampel berikut dalam MODE GURU. Kembalikan HANYA JSON Array.\n\n{json.dumps(samples, ensure_ascii=False)}"

    payload = {
        "model": "guru",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
    }

    for attempt in range(MAX_RETRY):
        try:
            res = requests.post(API_URL, json=payload, timeout=600).json()
            content = res["choices"][0]["message"]["content"]
            parsed = extract_json_array(content)

            if len(parsed) != len(samples):
                raise ValueError(f"Jumlah objek ({len(parsed)}) != jumlah sampel ({len(samples)})")

            with open(OUT_PATH, "a", encoding="utf-8") as f:
                for item in parsed:
                    idx = item.get("idx")
                    if idx in expected_golds:
                        item["label"] = expected_golds[idx] # Paksa label = gold
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        done_ids.add(idx)
            print(f"✅ Batch {i//BATCH + 1} selesai. (Progress: {len(done_ids)}/{len(rows)})")
            break
        except Exception as e:
            print(f"⚠️ Percobaan {attempt+1} gagal untuk batch {i//BATCH + 1}: {e}")
            if attempt < MAX_RETRY - 1:
                time.sleep(5)
