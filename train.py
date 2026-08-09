#!/usr/bin/env python3
"""Fine-tune Qwen3.5-9B-Defiant (Unsloth QLoRA) untuk klasifikasi Cognitive Presence.

Dataset : data/dataset_sft_sharegpt.jsonl (1080 sampel, format {messages:[...]})
System  : prompts/test_system.txt (rubrik penuh ~22KB) — sudah dibaked di dataset.
Output  : LoRA adapter + merged 16-bit + (opsional) GGUF Q8_0 untuk eval.py.

Pakai:
  venv/bin/python3 train.py --probe                 # load + ukur token + cetak marker, TANPA train
  venv/bin/python3 train.py                          # training penuh
  venv/bin/python3 train.py --gguf                   # training + export GGUF (lama)
  venv/bin/python3 train.py --max-seq 12288 --epochs 3 --lora-r 16
"""
import argparse
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/models/Qwen3.5-9B-Defiant")
    ap.add_argument("--data", default=str(HERE / "data" / "dataset_sft_sharegpt.jsonl"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "qwen35-9b-defiant"))
    ap.add_argument("--max-seq", type=int, default=12288)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--response-template", default="<|im_start|>assistant\n",
                    help="penanda mulai jawaban assistant (completion-only loss). "
                         "Jalankan --probe dulu untuk lihat marker exact lalu override bila perlu.")
    ap.add_argument("--probe", action="store_true",
                    help="load model + format dataset + cetak diagnostik, lalu KELUAR (tanpa train).")
    ap.add_argument("--gguf", action="store_true",
                    help="setelah training, export GGUF Q8_0 (butuh llama.cpp build; bisa lambat).")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel

    max_seq = args.max_seq

    # ---- 1. Load model 4-bit ----
    print(f"[*] Loading model: {args.model}  (4-bit, max_seq={max_seq})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq,
        dtype=None,           # auto: bf16 kalau GPU mendukung
        load_in_4bit=True,
    )

    # ---- 2. LoRA adapters ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )

    # ---- 3. Dataset → format pakai CHAT TEMPLATE NATIF model (jangan override) ----
    ds = load_dataset("json", data_files=str(args.data), split="train")
    print(f"[*] Dataset: {len(ds)} sampel")

    def fmt(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=False)
                 for m in batch["messages"]]
        return {"text": texts}

    ds = ds.map(fmt, batched=True, num_proc=4)

    # ---- PROBE: diagnostik lalu keluar ----
    if args.probe:
        print("\n" + "=" * 64)
        print("PROBE — diagnostik (tanpa training)")
        print("=" * 64)
        try:
            print("[*] " + model.get_nb_trainable_parameters())
        except Exception:
            tot = sum(p.numel() for p in model.parameters())
            tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[*] params total={tot:,} trainable={tr:,} ({100*tr/tot:.3f}%)")
        lens = sorted(len(tokenizer(t, add_special_tokens=False)["input_ids"])
                      for t in ds["text"])
        print(f"[*] token length: min={lens[0]} median={lens[len(lens)//2]} "
              f"mean={int(statistics.mean(lens))} max={lens[-1]}")
        n_over = sum(1 for L in lens if L > max_seq)
        flag = "⚠ TRUNCATION — naikkan --max-seq" if n_over else "✓ aman"
        print(f"[*] sampel > max_seq({max_seq}): {n_over}/{len(lens)}  {flag}")

        # marker assistant yang EXACT (buat response_template completion-only loss)
        s0 = ds["text"][0]
        idx = s0.find("<|im_start|>assistant")
        marker = s0[idx:idx + 80]
        print("\n--- marker assistant (repr, 80 char) — pakai ini buat --response-template ---")
        print(repr(marker))
        print("\n--- formatted sample 0 head (300 char) ---")
        print(s0[:300])
        print("=" * 64)
        print("Kirim output ini ke saya untuk konfirmasi max-seq + response-template,")
        print("lalu jalankan training tanpa --probe.")
        print("=" * 64)
        return

    # ---- 4. Trainer + completion-only loss ----
    from trl import SFTTrainer, SFTConfig
    from transformers import DataCollatorForCompletionOnlyLM

    collator = DataCollatorForCompletionOnlyLM(
        response_template=args.response_template, tokenizer=tokenizer)

    cfg = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        report_to="none",
        max_seq_length=max_seq,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=collator,
        args=cfg,
    )

    print("[*] Mulai training...")
    trainer.train()

    # ---- 5. Save ----
    os.makedirs(args.out, exist_ok=True)
    adapter_dir = str(Path(args.out) / "lora")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[+] LoRA adapter: {adapter_dir}")

    merged_dir = str(Path(args.out) / "merged_16bit")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[+] Merged 16-bit: {merged_dir}")

    if args.gguf:
        gguf_dir = str(Path(args.out) / "gguf")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q8_0")
        print(f"[+] GGUF Q8_0: {gguf_dir}")

    print("\n[+] SELESAI. Untuk evaluasi: load GGUF via llama-server → jalankan eval.py.")


if __name__ == "__main__":
    main()
