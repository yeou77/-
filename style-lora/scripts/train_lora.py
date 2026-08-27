"""
QLoRA fine-tune to make a base model pick up an author's rhythm/vocabulary
from data/processed/train_corpus.txt (produced by preprocess.py build).

This is plain causal-LM continued pretraining on raw narrative text (no
instruction format) — the goal is style/rhythm, not chat behavior.

Example (on a single ~16-24GB cloud GPU):
    python train_lora.py \
        --model_name beomi/Llama-3-Open-Ko-8B-Instruct-preview \
        --train_file ../data/processed/train_corpus.txt \
        --output_dir ../outputs/geoneomulnyeo-lora

Check the base model's license before using the adapter beyond personal use.
"""
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def build_dataset(tokenizer, train_file: str, block_size: int) -> Dataset:
    text = Path(train_file).read_text(encoding="utf-8")
    ids = tokenizer(text, return_attention_mask=False)["input_ids"]

    n_blocks = len(ids) // block_size
    if n_blocks == 0:
        raise SystemExit(
            f"코퍼스가 block_size({block_size}) 토큰보다 짧습니다. "
            "더 작은 --block_size를 쓰거나 데이터를 늘리세요."
        )
    ids = ids[: n_blocks * block_size]
    blocks = [ids[i : i + block_size] for i in range(0, len(ids), block_size)]
    return Dataset.from_dict({"input_ids": blocks, "labels": [b[:] for b in blocks]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", default="beomi/Llama-3-Open-Ko-8B-Instruct-preview")
    ap.add_argument("--train_file", default="../data/processed/train_corpus.txt")
    ap.add_argument("--output_dir", default="../outputs/style-lora")
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("코퍼스 토큰화 중...")
    dataset = build_dataset(tokenizer, args.train_file, args.block_size)
    print(f"학습 블록 수: {len(dataset)} (block_size={args.block_size})")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA 어댑터 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
