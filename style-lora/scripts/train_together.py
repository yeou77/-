"""
Fine-tunes a LoRA on Together.ai instead of a local/Colab GPU. Run this from
anywhere with normal internet access (a Colab cell is fine — no GPU runtime
needed here, Together's cloud does the actual training).

Needs data/processed/train_corpus.txt to already exist (run
`preprocess.py auto` first).

Usage:
    python train_together.py --api-key-file /path/to/key.txt \
        --model Qwen/Qwen3.5-9B

This is a first pass against Together's API — if a call fails, the raw
error response is printed so the exact field name/format can be fixed
without guessing blind.
"""
import argparse
import json
import sys
import time
from pathlib import Path

TRAIN_CORPUS = Path(__file__).resolve().parent.parent / "data" / "processed" / "train_corpus.txt"
JSONL_OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "together_train.jsonl"


def build_jsonl(chunk_chars: int) -> int:
    text = TRAIN_CORPUS.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > chunk_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)

    with JSONL_OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
    return len(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-key-file", required=True, help="Together API 키가 담긴 파일 경로")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--chunk-chars", type=int, default=3000)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--download-to", default="../outputs/style-lora-together.tar.zst")
    args = ap.parse_args()

    try:
        from together import Together
    except ImportError:
        sys.exit("pip install together 먼저 실행하세요.")

    if not TRAIN_CORPUS.exists():
        sys.exit(f"{TRAIN_CORPUS} 가 없습니다. 먼저 preprocess.py auto를 실행하세요.")

    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    client = Together(api_key=api_key)

    n_chunks = build_jsonl(args.chunk_chars)
    print(f"학습용 JSONL {n_chunks}개 청크 -> {JSONL_OUT}")

    print("파일 업로드 중...")
    file_resp = client.files.upload(file=str(JSONL_OUT), check=True)
    file_id = file_resp.id
    print(f"업로드 완료: file_id={file_id}")

    print("파인튜닝 job 생성 중...")
    ft_resp = client.fine_tuning.create(
        training_file=file_id,
        model=args.model,
        n_epochs=args.epochs,
        lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        suffix="style-lora",
    )
    job_id = ft_resp.id
    print(f"job 생성됨: job_id={job_id} (Together 대시보드 Fine-tuning 탭에서도 진행률 확인 가능)")

    print("진행 상황 확인 중 (Ctrl+C로 중단해도 job 자체는 계속 돌아갑니다)...")
    while True:
        status = client.fine_tuning.retrieve(job_id)
        print(f"  status={status.status}")
        if status.status in ("completed", "error", "cancelled", "failed"):
            break
        time.sleep(args.poll_seconds)

    if status.status != "completed":
        print(f"\n학습이 정상 종료되지 않았습니다: {status}")
        sys.exit(1)

    print("\n학습 완료. 어댑터 다운로드 시도...")
    out_path = Path(args.download_to)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    download_fn = None
    for attr_name in ("download", "download_model", "download_checkpoint"):
        candidate = getattr(client.fine_tuning, attr_name, None)
        if callable(candidate):
            download_fn = candidate
            break

    if download_fn is None:
        available = [m for m in dir(client.fine_tuning) if not m.startswith("_")]
        print("다운로드 함수를 SDK에서 찾지 못했습니다 (RUN_LOG.md 이식성 체크리스트 참고).")
        print(f"client.fine_tuning에 실제로 있는 메서드: {available}")
        print(f"job_id={job_id} 는 기록해두고, Together 대시보드에서 수동 다운로드 가능한지 확인하세요.")
        return

    try:
        download_fn(id=job_id, output=str(out_path))
        print(f"다운로드 완료: {out_path}")
    except Exception as e:
        print(f"다운로드 실패: {e}")
        print(f"job_id={job_id} 는 기록해두고, Together 대시보드에서 수동 다운로드 가능한지 확인하세요.")


if __name__ == "__main__":
    main()
