"""
Splits raw novel .txt files into chapters/paragraphs and builds a training
corpus with a small evaluation set automatically held out.

Easiest path — one command, no manual picking:
    # drop book1.txt, book2.txt (full novels) into data/raw/, then:
    python preprocess.py auto

That writes data/processed/train_corpus.txt and eval_set.jsonl directly,
with the eval paragraphs chosen by random sample (not by hand).

If you'd rather hand-pick which paragraphs represent the style best, use
the two-step version instead:
    python preprocess.py candidates
    # open data/processed/eval_candidates.jsonl, pick ~30 ids you like,
    # save them one per line into eval_selected.txt
    python preprocess.py build --eval-ids eval_selected.txt
"""
import argparse
import json
import random
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CHAPTER_RE = re.compile(
    r"^\s*(제?\s*\d+\s*화|chapter\s*\d+|\d+\s*장|\d+\.\s*$)", re.IGNORECASE
)

DIALOGUE_MARKS = ('"', "“", "”", "'", "‘", "’")
PSYCH_HINTS = ("싶었다", "생각했다", "생각이 들었다", "것 같았다", "느꼈다", "싶어졌다")


def split_chapters(text: str) -> list[str]:
    lines = text.splitlines()
    boundaries = [i for i, ln in enumerate(lines) if CHAPTER_RE.match(ln)]
    if len(boundaries) < 2:
        # no reliable chapter markers -> split on runs of 3+ blank lines
        chunks = re.split(r"\n{3,}", text)
        return [c.strip() for c in chunks if c.strip()]
    boundaries.append(len(lines))
    chapters = []
    for start, end in zip(boundaries, boundaries[1:]):
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            chapters.append(chunk)
    return chapters


def split_paragraphs(chapter_text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", chapter_text)
    return [p.strip() for p in paras if p.strip() and len(p.strip()) > 5]


def tag_paragraph(p: str) -> str:
    dialogue_chars = sum(p.count(c) for c in DIALOGUE_MARKS)
    if dialogue_chars >= 2:
        return "대사"
    if any(hint in p for hint in PSYCH_HINTS):
        return "심리"
    return "지문"


def collect_paragraphs() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    books = sorted(RAW_DIR.glob("*.txt"))
    if not books:
        raise SystemExit(f"data/raw/ 에 .txt 파일이 없습니다: {RAW_DIR}")

    all_paragraphs = []
    for book_path in books:
        book_id = book_path.stem
        text = book_path.read_text(encoding="utf-8")
        chapters = split_chapters(text)
        chap_dir = OUT_DIR / book_id / "chapters"
        chap_dir.mkdir(parents=True, exist_ok=True)

        pid = 0
        for ci, chap in enumerate(chapters):
            (chap_dir / f"{ci:04d}.txt").write_text(chap, encoding="utf-8")
            for para in split_paragraphs(chap):
                all_paragraphs.append(
                    {
                        "id": f"{book_id}:{ci:04d}:{pid}",
                        "book": book_id,
                        "chapter": ci,
                        "tag": tag_paragraph(para),
                        "len": len(para),
                        "text": para,
                    }
                )
                pid += 1
        print(f"[{book_id}] {len(chapters)} chapters, {pid} paragraphs")

    para_path = OUT_DIR / "paragraphs.jsonl"
    with para_path.open("w", encoding="utf-8") as f:
        for p in all_paragraphs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n총 {len(all_paragraphs)}개 문단 -> {para_path}")
    return all_paragraphs


def write_train_and_eval(all_paragraphs: list[dict], eval_ids: set[str]) -> None:
    train_lines = [p["text"] for p in all_paragraphs if p["id"] not in eval_ids]
    eval_paragraphs = [p for p in all_paragraphs if p["id"] in eval_ids]

    train_corpus_path = OUT_DIR / "train_corpus.txt"
    train_corpus_path.write_text("\n\n".join(train_lines), encoding="utf-8")

    eval_set_path = OUT_DIR / "eval_set.jsonl"
    with eval_set_path.open("w", encoding="utf-8") as f:
        for p in eval_paragraphs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"학습 코퍼스: {len(train_lines)}개 문단 -> {train_corpus_path}")
    print(f"평가 세트: {len(eval_paragraphs)}개 문단 (학습에서 제외됨) -> {eval_set_path}")


def cmd_auto(args):
    all_paragraphs = collect_paragraphs()

    rng = random.Random(args.seed)
    by_book: dict[str, list[dict]] = {}
    for p in all_paragraphs:
        by_book.setdefault(p["book"], []).append(p)

    # spread the eval sample evenly across books so no single book dominates it
    eval_ids: set[str] = set()
    n_books = len(by_book)
    per_book = max(1, args.eval_n // n_books)
    for book_paragraphs in by_book.values():
        sample = rng.sample(book_paragraphs, min(per_book, len(book_paragraphs)))
        eval_ids.update(p["id"] for p in sample)

    write_train_and_eval(all_paragraphs, eval_ids)
    print("(평가 문단은 무작위로 골랐습니다 — 직접 고르고 싶으면 candidates/build를 대신 쓰세요.)")


def cmd_candidates(args):
    all_paragraphs = collect_paragraphs()

    # sample candidates per (book, tag) so the user has a manageable list to read
    by_key: dict[tuple[str, str], list[dict]] = {}
    for p in all_paragraphs:
        by_key.setdefault((p["book"], p["tag"]), []).append(p)

    candidates = []
    for key, group in by_key.items():
        group_sorted = sorted(group, key=lambda x: x["len"], reverse=True)
        candidates.extend(group_sorted[: args.per_group])

    cand_path = OUT_DIR / "eval_candidates.jsonl"
    with cand_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"후보 {len(candidates)}개 (book x tag별 상위 {args.per_group}개) -> {cand_path}")
    print("이 중 ~30개를 골라 id만 한 줄씩 eval_selected.txt 에 저장하세요.")


def cmd_build(args):
    para_path = OUT_DIR / "paragraphs.jsonl"
    if not para_path.exists():
        raise SystemExit("먼저 `python preprocess.py candidates` 를 실행하세요.")

    eval_ids_path = Path(args.eval_ids)
    eval_ids = {
        line.strip()
        for line in eval_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not eval_ids:
        raise SystemExit(f"{eval_ids_path} 에서 평가용 id를 읽지 못했습니다.")

    all_paragraphs = [json.loads(line) for line in para_path.open(encoding="utf-8")]
    missing = eval_ids - {p["id"] for p in all_paragraphs}
    if missing:
        print(f"경고: paragraphs.jsonl에서 못 찾은 id {len(missing)}개: {sorted(missing)[:5]} ...")

    write_train_and_eval(all_paragraphs, eval_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_auto = sub.add_parser("auto", help="원클릭: 분리 + 무작위 평가셋 + 학습 코퍼스까지 한 번에")
    p_auto.add_argument("--eval-n", type=int, default=20, help="평가용으로 뗄 문단 총 개수")
    p_auto.add_argument("--seed", type=int, default=42)
    p_auto.set_defaults(func=cmd_auto)

    p_cand = sub.add_parser("candidates", help="챕터/문단 분리 + 평가 후보 추출 (수동 선별용)")
    p_cand.add_argument("--per-group", type=int, default=15)
    p_cand.set_defaults(func=cmd_candidates)

    p_build = sub.add_parser("build", help="평가 세트를 제외한 학습 코퍼스 생성")
    p_build.add_argument("--eval-ids", required=True, help="선택한 paragraph id 목록 txt")
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
