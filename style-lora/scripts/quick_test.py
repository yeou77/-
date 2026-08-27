"""
Loads a base model + a local LoRA adapter directory and generates a
continuation for one prompt. For a fast sanity check ("does the style show
up at all") without needing the full eval.py pipeline or an external
service's hosted inference.

Usage:
    python quick_test.py --base_model Qwen/Qwen3.5-9B \
        --adapter_dir outputs/style-lora-together \
        --prompt "그는 문을 열었다."
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("=== +LoRA ===")
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    with torch.no_grad(), model.disable_adapter():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("\n=== base (비교용, 어댑터 비활성화) ===")
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
