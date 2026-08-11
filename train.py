#!/usr/bin/env python3
"""Fine-tune Qwen3.5-9B-Defiant (Unsloth FastVisionModel, LoRA) untuk klasifikasi Cognitive Presence.

Qwen3.5 = VLM unified (Causal LM + Vision Encoder). Meski dataset kita TEXT-ONLY,
WAJIB pakai FastVisionModel (bukan FastLanguageModel): tokenizernya adalah VL
processor, jadi tokenizer(text) di FastLanguageModel error (routes ke image processor).
Vision encoder TIDAK di-train (finetune_vision_layers=False) — hanya LM yang di-train.

Method (per doc resmi Unsloth Qwen3.5):
  --lora  (DEFAULT) base 16-bit + adapter LoRA   ← RECOMMENDED (QLoRA NOT recommended utk Qwen3.5)
  --qlora           base 4-bit + adapter LoRA    ← flag disediakan buat model lain; di-avoid utk Qwen3.5

Dataset : data/dataset_sft_sharegpt.jsonl (1080 sampel, format {messages:[role/content]})
Output  : LoRA adapter + merged 16-bit + GGUF Q8_0 (otomatis) untuk eval.py.

Pakai:
  venv/bin/python3 train.py --probe                 # load + ukur token + test collator, TANPA train
  venv/bin/python3 train.py                          # LoRA 16-bit (DEFAULT) + auto Q8_0 GGUF
  venv/bin/python3 train.py --qlora                  # QLoRA 4-bit (hemat VRAM; bukan utk Qwen3.5)
  venv/bin/python3 train.py --skip-gguf              # training saja (skip export GGUF yang lambat)
  venv/bin/python3 train.py --max-seq 12288 --epochs 3 --lora-r 16
"""
import argparse
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
                    help="penanda mulai jawaban assistant (completion-only loss via "
                         "UnslothVisionDataCollator train_on_responses_only).")
    method = ap.add_mutually_exclusive_group()
    method.add_argument("--lora", action="store_true",
                        help="(DEFAULT) base 16-bit full + adapter LoRA. Buat GPU besar (mis. A100 80GB).")
    method.add_argument("--qlora", action="store_true",
                        help="base 4-bit + adapter LoRA. Hemat VRAM. "
                             "[Catatan: QLoRA NOT recommended utk Qwen3.5 per doc resmi Unsloth.]")
    ap.add_argument("--probe", action="store_true",
                    help="load model + dataset + build trainer + test collator, lalu KELUAR (tanpa train).")
    ap.add_argument("--skip-gguf", action="store_true",
                    help="lewati export GGUF Q8_0 (DEFAULT: otomatis setelah training).")
    ap.add_argument("--scratch", default=None,
                    help="dir cepat (mis. /dev/shm) buat merged_16bit + gguf. Default = --out. "
                         "Model besar (35B): route merge 66GB ke RAM (hindari mfs lambat/ENOSPC).")
    args = ap.parse_args()
    scratch_dir = args.scratch or args.out
    load_4bit = args.qlora                     # default LoRA (16-bit); --qlora -> 4-bit
    method_name = "QLoRA (4-bit)" if load_4bit else "LoRA (16-bit)"

    import torch
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    max_seq = args.max_seq

    # ---- 1. Load VLM (FastVisionModel — bukan FastLanguageModel) ----
    # Qwen3.5 = VLM: tokenizer returned = full VL processor.
    print(f"[*] Method: {method_name} | Loading: {args.model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        dtype=None,                       # auto: bf16 di A100
        load_in_4bit=load_4bit,
        use_gradient_checkpointing="unsloth",   # JANGAN ganti ke True! Mode "unsloth" = smart offload (recompute aktivasi LEBIH SEDIKIT) -> 3x lebih cepat dari mode True (full recompute). Offload CPU↔GPU di-double-buffer, parallel dgn compute -> bukan bottleneck.
    )

    # ---- 2. LoRA — TEXT-ONLY: vision encoder OFF, language/attention/mlp ON ----
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # text-only: jangan train vision encoder
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    FastVisionModel.for_training(model)

    # ---- 3. Dataset: messages mentah (collator yang render+tokenize via processor) ----
    ds = load_dataset("json", data_files=str(args.data), split="train")
    print(f"[*] Dataset: {len(ds)} sampel")

    # ---- 4. Collator (completion-only loss bawaan) + Trainer ----
    collator = UnslothVisionDataCollator(
        model, tokenizer,
        max_seq_length=max_seq,        # WAJIB: tanpa ini default ke model.max_seq_length (2048) -> TRUNCATE rubrik!
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part=args.response_template,
    )
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
        # VLM-required: jangan tokenize text-column; collator yang handle messages.
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_seq,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,        # full processor (Unsloth VLM butuh tokenizer=, bukan processing_class=)
        train_dataset=ds,
        data_collator=collator,
        args=cfg,
    )

    # ---- PROBE ----
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

        # token length via chat template. Robust: kalau VL processor ngambek di tokenize=True,
        # fallback ke estimasi char-based (tokenize=False udah terbukti jalan).
        def n_tokens(msgs):
            try:
                enc = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False, return_dict=False)
                ids = enc["input_ids"] if isinstance(enc, dict) else enc
                return len(ids)
            except Exception:
                return len(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)) // 3

        lens = sorted(n_tokens(m) for m in ds["messages"])
        print(f"[*] token length: min={lens[0]} median={lens[len(lens)//2]} "
              f"mean={int(statistics.mean(lens))} max={lens[-1]}")
        n_over = sum(1 for L in lens if L > max_seq)
        flag = "⚠ TRUNCATION — naikkan --max-seq" if n_over else "✓ aman"
        print(f"[*] sampel > max_seq({max_seq}): {n_over}/{len(lens)}  {flag}")

        s0 = tokenizer.apply_chat_template(
            ds["messages"][0], tokenize=False, add_generation_prompt=False)
        idx = s0.find("<|im_start|>assistant")
        print("\n--- marker assistant (repr, 80 char) ---")
        print(repr(s0[idx:idx + 80]))

        # test collator di data text-only (validasi path data beneran jalan sebelum training mahal)
        try:
            batch = collator([ds[i] for i in range(min(2, len(ds)))])
            print(f"\n[*] collator max_seq_length={collator.max_seq_length} | "
                  f"input_ids shape={batch['input_ids'].shape}")
        except Exception as e:
            print(f"\n[!] collator test GAGAL: {e}")
            print("[!] Collator mungkin butuh key 'images' — kirim output ini ke saya.")
        print("=" * 64)
        return

    # ---- 5. Train ----
    print("[*] Mulai training...")
    trainer.train()

    # ---- 6. Save: LoRA + merged 16-bit + GGUF Q8_0 (otomatis) ----
    os.makedirs(args.out, exist_ok=True)
    adapter_dir = str(Path(args.out) / "lora")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[+] LoRA adapter: {adapter_dir}")

    merged_dir = str(Path(scratch_dir) / "merged_16bit")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[+] Merged 16-bit (sumber konversi GGUF): {merged_dir}")

    if not args.skip_gguf:
        gguf_dir = str(Path(scratch_dir) / "gguf")
        print("[*] Export GGUF Q8_0 (otomatis, ~10-30 menit)...")
        try:
            # API terverifikasi (unsloth.ai/docs saving-to-gguf):
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q8_0")
            print(f"[+] GGUF Q8_0: {gguf_dir}")
        except Exception as e:
            # ponytail: save_pretrained_gguf paling sering gagal di "llama.cpp not found";
            # fallback manual pakai build llama.cpp yang sudah ada di repo pod.
            print(f"[!] Export otomatis GAGAL: {e}")
            print("[!] Fallback manual (jalankan sendiri, sesuaikan path build bila beda):")
            print(f"      python {HERE}/llama.cpp/convert_hf_to_gguf.py "
                  f"{merged_dir} --outfile {merged_dir}/model.f16.gguf --outtype f16")
            print(f"      {HERE}/llama.cpp/build/bin/llama-quantize "
                  f"{merged_dir}/model.f16.gguf {merged_dir}/model-q8_0.gguf q8_0")

    print("\n[+] SELESAI. Untuk evaluasi: load GGUF Q8_0 via llama-server → jalankan eval.py.")


if __name__ == "__main__":
    main()
