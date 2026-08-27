"""
Generates continuations from the base model and the base+LoRA model for the
same prompts, so you can read them side by side and judge which one sounds
like the target author. There is no automatic score here on purpose — per
the plan, your own read of the output is the benchmark, not a number.

Prompts come from data/processed/eval_set.jsonl (the paragraphs you held out
in preprocess.py build): each paragraph's first sentence is used as the
prompt, and the model has to continue it.

Usage:
    python eval.py --lora_dir ../outputs/style-lora --out ../outputs/eval_report.md
"""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval_set.jsonl"


def first_sentence(text: str) -> str:
    m = re.search(r".+?[.!?…”]\s", text)
    return (m.group(0) if m else text[:60]).strip()


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", default="beomi/Llama-3-Open-Ko-8B-Instruct-preview")
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--n", type=int, default=8, help="평가 세트에서 몇 개 프롬프트를 뽑을지")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--out", default="../outputs/eval_report.md")
    args = ap.parse_args()

    if not EVAL_SET.exists():
        raise SystemExit(f"{EVAL_SET} 가 없습니다. 먼저 preprocess.py build를 실행하세요.")

    paragraphs = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines()]
    prompts = [(p["id"], first_sentence(p["text"]), p["text"]) for p in paragraphs[: args.n]]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto"
    )
    lora_model = PeftModel.from_pretrained(base_model, args.lora_dir)

    report = ["# Style LoRA eval report\n"]
    for pid, prompt, gold in prompts:
        base_out = generate(base_model, tokenizer, prompt, args.max_new_tokens)
        lora_out = generate(lora_model, tokenizer, prompt, args.max_new_tokens)

        report.append(f"## {pid}\n")
        report.append(f"**prompt**: {prompt}\n")
        report.append(f"**원문 (참고용, 정답 아님)**:\n> {gold}\n")
        report.append(f"**base**:\n> {base_out}\n")
        report.append(f"**+lora**:\n> {lora_out}\n")
        report.append("---\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"리포트 저장: {out_path}")


if __name__ == "__main__":
    main()
