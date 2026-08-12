#!/usr/bin/env python3
"""Benchmark LLM (llama-server) untuk klasifikasi Community of Inquiry (CoI).

Mengirim setiap postingan test set ke endpoint OpenAI-compatible /v1/chat/completions
pada llama.cpp, lalu menilai (grade) prediksi terhadap label emas.

Pakai: venv/bin/python3 eval.py --prompt few --host 127.0.0.1 --port 8889
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

LABELS = ["C0", "C1", "C2", "C3", "C4"]
CLASS_NAMES = [
    "Bukan Kehadiran Kognitif",
    "Pemantik Diskusi",
    "Eksplorasi",
    "Integrasi",
    "Resolusi",
]
HERE = Path(__file__).resolve().parent
SCHEMA = {
    "type": "object",
    "properties": {
        "alasan": {"type": "string"},
        "label": {"type": "string", "enum": LABELS},
    },
    "required": ["alasan", "label"],
    "additionalProperties": False,
}
# ponytail: format response_format milik llama.cpp = {"type":..,"schema":..}
# (berbeda dari OpenAI yang nested di bawah "json_schema")
FORMAT_OPTIONS = [
    ("json_schema", {"type": "json_schema", "schema": SCHEMA}),
    ("json_object", {"type": "json_object"}),
    ("none", None),
]
# ponytail: sampling (temp/top_p/top_k) tidak di-override di sini — pakai default server.


def load_test(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("text") and r.get("label")]
    return rows


def build_messages(system, user_tpl, text):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_tpl.replace("{text}", text)},
    ]


def server_props(base):
    r = requests.get(f"{base}/props", timeout=10)
    r.raise_for_status()
    return r.json()


def probe_format(base, system, user_tpl):
    """Cari mode response_format tertinggi yang menghasilkan jawaban TERPARSE.
    Memverifikasi output benar-benar punya label valid — bukan cuma HTTP 200 —
    karena reasoning model bisa terpotong di tengah blok <think> sehingga tak ada jawaban."""
    msgs = build_messages(system, user_tpl, "halo")
    for name, rf in FORMAT_OPTIONS:
        body = {"model": "local", "messages": msgs, "seed": 42,
                "max_tokens": 1024, "stream": False}
        if rf:
            body["response_format"] = rf
        try:
            r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=120)
        except requests.RequestException:
            continue
        if not r.ok:
            continue
        msg = r.json()["choices"][0]["message"]
        answer, _ = split_think_answer(msg.get("content"), msg.get("reasoning_content"))
        if parse_label(answer)[0]:
            return name
    return "none"


def call_llm(base, messages, fmt, max_tokens, timeout):
    rf = dict(FORMAT_OPTIONS).get(fmt)
    body = {"model": "local", "messages": messages, "seed": 42,
            "max_tokens": max_tokens, "stream": False}
    if rf:
        body["response_format"] = rf
    t0 = time.time()
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=timeout)
    latency = (time.time() - t0) * 1000
    r.raise_for_status()
    choice = r.json()["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish = choice.get("finish_reason", "")
    return content, reasoning, finish, latency


def split_think_answer(content, reasoning):
    """Pisahkan blok thinking dari jawaban final (untuk model reasoning).
    Menerima field content dan reasoning_content dari llama.cpp. Menangani reasoning
    yang sudah terpisah ke field sendiri MAUPUN yang masih inline <think>...</think>.
    Return (answer, thinking)."""
    thinking = reasoning or ""
    answer = content or ""
    m = re.search(r"<think>(.*?)</think>(.*)", answer, re.DOTALL)
    if m:
        thinking = (thinking + "\n" + m.group(1)).strip()
        answer = m.group(2).strip()
    return answer, thinking


def parse_label(content):
    """Ambil label dari output. Return (label|None, reason|None)."""
    if not content.strip():
        return None, None
    # 1) JSON utuh
    try:
        data = json.loads(content)
        lab = str(data.get("label", "")).strip().upper()
        if lab in LABELS:
            return lab, str(data.get("alasan", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    # 2) JSON terkubung di tengah teks
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            lab = str(data.get("label", "")).strip().upper()
            if lab in LABELS:
                return lab, str(data.get("alasan", "")).strip()
        except json.JSONDecodeError:
            pass
    # 3) regex label mentah (fallback terakhir)
    m = re.search(r"\bC[0-4]\b", content)
    if m:
        return m.group(0), None
    return None, None


def erase_slot(base, slot=0):
    try:
        requests.post(f"{base}/slots/{slot}?action=erase", timeout=10)
    except requests.RequestException:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", choices=["few", "zero"], default="few",
                    help="few = few-shot (definisi + contoh + penjelasan tambahan); "
                         "zero = zero-shot (definisi kelas saja, tanpa contoh/penjelasan tambahan)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--data", default=str(HERE / "data" / "test_dataset.csv"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="budget token per sampel (cukup untuk thinking+jawaban)")
    ap.add_argument("--max-budget", type=int, default=8192,
                    help="batas atas budget saat auto-escalate truncation")
    ap.add_argument("--limit", type=int, default=0, help="0 = semua 120 sampel")
    ap.add_argument("--erase", action="store_true",
                    help="POST /slots/0?action=erase antar sampel (ekstra hati-hati)")
    ap.add_argument("--dry-run", action="store_true",
                    help="cetak 1 prompt lalu keluar (tanpa hubungi server)")
    ap.add_argument("--fresh", action="store_true",
                    help="abaikan checkpoint, mulai dari awal")
    args = ap.parse_args()

    system = (HERE / "prompts" / f"system_{args.prompt}_shot.md").read_text(encoding="utf-8")
    user_tpl = (HERE / "prompts" / "user.md").read_text(encoding="utf-8")
    rows = load_test(args.data)
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        print("=== SYSTEM ===\n", system)
        print("\n=== USER (sampel 0) ===\n", build_messages(system, user_tpl, rows[0]["text"])[1]["content"])
        return

    base = f"http://{args.host}:{args.port}"
    print(f"[*] Server: {base}")
    try:
        props = server_props(base)
    except requests.RequestException as e:
        sys.exit(f"[!] Tidak bisa hubungi {base}/props: {e}\n    "
                 f"Mulai server: llama-server -m <model.gguf> --host 0.0.0.0 --port {args.port} "
                 f"-c 4096 -ngl 99")
    model = props.get("model_path", "?")
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx", "?")
    print(f"[*] Model: {model}  |  n_ctx: {n_ctx}  |  slots: {props.get('total_slots')}")

    fmt = probe_format(base, system, user_tpl)
    print(f"[*] response_format terbaik: {fmt}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.prompt}shot"
    ckpt = out / f"_ckpt_{tag}.jsonl"
    # --- resume: muat hasil yang sudah selesai dari checkpoint ---
    done = {}
    if ckpt.exists() and not args.fresh:
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["idx"]] = rec
        print(f"[*] RESUME: {len(done)} sampel sudah ada, lanjut sisanya")
    elif args.fresh and ckpt.exists():
        ckpt.unlink()
        print("[*] --fresh: checkpoint dihapus, mulai dari awal")
    ckpt_fh = open(ckpt, "a", encoding="utf-8")

    preds, golds, records = [], [], []
    n_fail = n_trunc = 0
    t_start = time.time()
    for i, row in enumerate(rows):
        text, gold = row["text"], row["label"]
        if i in done:
            rec = done[i]
            preds.append(rec["pred"]); golds.append(rec["gold"]); records.append(rec)
            print(f"  [{i+1:3d}/{len(rows)}] \u23ed skip (resume) gold={rec['gold']} pred={rec['pred']}")
            continue
        msgs = build_messages(system, user_tpl, text)
        pred = reason = answer = thinking = ""
        latency, err, finish, budget = 0.0, "", "", args.max_tokens
        for attempt in range(4):
            try:
                raw, reasoning, finish, latency = call_llm(
                    base, msgs, fmt, budget, timeout=300)
            except requests.RequestException as e:
                err = str(e); time.sleep(2 ** attempt); continue
            answer, thinking = split_think_answer(raw, reasoning)
            pred, reason = parse_label(answer)
            if not pred and thinking:
                # ponytail: model kadang taruh JSON final di reasoning_content,
                # content kosong. Fallback parse dari thinking biar gak PARSE_FAIL.
                pred, reason = parse_label(thinking)
                if pred:
                    answer = thinking   # simpan sumber label utk answer_head
            if pred:
                break
            # terpotong di tengah thinking? beri ruang lebih lalu ulang
            if finish == "length" and budget < args.max_budget:
                n_trunc += 1
                budget = min(budget * 2, args.max_budget)
                err = f"truncated, retry budget={budget}"
                continue
            err = f"tidak terparse (finish={finish}, attempt {attempt+1})"
        if not pred:
            n_fail += 1
            pred = "PARSE_FAIL"
        preds.append(pred); golds.append(gold)
        rec = {
            "idx": i, "text": text[:500], "gold": gold, "pred": pred,
            "reason": reason, "latency_ms": round(latency, 1),
            "answer_head": answer[:160], "thinking_head": thinking[:500],
            "finish": finish, "budget": budget, "error": err,
        }
        records.append(rec)
        # checkpoint: tulis+flush tiap sampel (resume zero-loss)
        ckpt_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ckpt_fh.flush()
        ok = "✓" if pred == gold else "✗"
        print(f"  [{i+1:3d}/{len(rows)}] {ok} gold={gold} pred={pred} "
              f"({latency:.0f}ms, think={len(thinking)}ch) {err}")
        if args.erase:
            erase_slot(base)
        if (i + 1) % 10 == 0:
            v = [(p, g) for p, g in zip(preds, golds) if p in LABELS]
            pacc = sum(1 for p, g in v if p == g) / len(v) if v else 0.0
            (out / f"_progress_{tag}.json").write_text(json.dumps({
                "done": len(records), "total": len(rows),
                "partial_accuracy": round(pacc, 4),
                "n_fail": n_fail, "n_trunc": n_trunc,
                "elapsed_s": round(time.time() - t_start, 1),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"      \u21b3 checkpoint@{len(records)}/{len(rows)} (partial acc={pacc:.3f})")

    elapsed = time.time() - t_start

    # --- Metrik ---
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [(p, g) for p, g in zip(preds, golds) if p in LABELS]
    v_preds = [p for p, _ in valid]
    v_golds = [g for _, g in valid]
    acc = accuracy_score(v_golds, v_preds) if valid else 0.0
    macro_f1 = f1_score(v_golds, v_preds, average="macro", labels=LABELS,
                        zero_division=0) if valid else 0.0
    weighted_f1 = f1_score(v_golds, v_preds, average="weighted", labels=LABELS,
                           zero_division=0) if valid else 0.0
    # strict: parse-fail dihitung salah
    acc_strict = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)

    report = classification_report(v_golds, v_preds, labels=LABELS,
                                   target_names=CLASS_NAMES, digits=4,
                                   zero_division=0)
    cm = confusion_matrix(v_golds, v_preds, labels=LABELS)

    with open(out / f"report_{tag}.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Prompt variant : {args.prompt}-shot\n")
        fh.write(f"Model          : {model}\n")
        fh.write(f"n_ctx          : {n_ctx}\n")
        fh.write(f"response_format: {fmt}\n")
        fh.write(f"Sampling       : default server (tidak di-override), seed=42\n")
        fh.write(f"max_tokens     : {args.max_tokens} (budget cap {args.max_budget})\n")
        fh.write(f"Sampel         : {len(rows)} (parse_fail={n_fail}, truncated_retry={n_trunc})\n")
        fh.write(f"Total waktu    : {elapsed:.1f}s ({elapsed/len(rows):.1f}s/sampel)\n")
        fh.write(f"Accuracy (valid):   {acc:.4f}\n")
        fh.write(f"Accuracy (strict):  {acc_strict:.4f}  [parse-fail=salah]\n")
        fh.write(f"Macro-F1:           {macro_f1:.4f}\n")
        fh.write(f"Weighted-F1:        {weighted_f1:.4f}\n\n")
        fh.write(report)

    with open(out / f"predictions_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: r["idx"]):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ckpt_fh.close()
    if ckpt.exists():
        ckpt.unlink()  # sukses lengkap -> hapus checkpoint resume
    prog = out / f"_progress_{tag}.json"
    if prog.exists():
        prog.unlink()

    # confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Label Emas")
    ax.set_title(f"Confusion Matrix — {tag} (acc={acc:.3f})")
    for yi in range(5):
        for xi in range(5):
            ax.text(xi, yi, cm[yi, xi], ha="center", va="center",
                    color="white" if cm[yi, xi] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out / f"confusion_matrix_{tag}.png", dpi=120)
    print(f"\n[+] Selesai. Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | "
          f"strict={acc_strict:.4f} | fail={n_fail}/{len(rows)} | {elapsed:.0f}s")
    print(f"[+] Output: {out}/  (report_{tag}.txt, predictions_{tag}.jsonl, "
          f"confusion_matrix_{tag}.png)")


if __name__ == "__main__":
    main()
