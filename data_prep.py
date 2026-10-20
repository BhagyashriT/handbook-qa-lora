"""Build the labeled topic-classification dataset from the handbook corpus.

Pipeline:
  markdown files -> markdown-aware chunks (vendored from rag-handbook-qa)
  -> label = top-level directory (weak supervision)
  -> filter tiny/meta chunks
  -> stratified FILE-level train/eval split (no file appears in both splits,
     so near-duplicate overlapping windows can't leak across the split)

Label curation (documented, deterministic):
  - Drop root-level meta files: README.md, CONTRIBUTING.md, PUBLISHING.md,
    RESOURCES.md (not interview content).
  - Drop src/ (a single README stub).
  - Merge practice/ + quick-revision/ -> "prep" (both are interview-prep
    artifacts: question banks, study plans, revision sheets; too few files
    each to stand alone as classes).
  - Drop chunks shorter than MIN_CHARS (mostly bare headers).

Final label set: backend, behavioral, dsa, java, learnings, lld, prep,
system-design (8 classes).

Usage:
  python data_prep.py --corpus /path/to/java-backend-interview-handbook \
      --out data [--max-chars 900] [--overlap 120] [--eval-frac 0.2] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from hqlora.chunking import chunk_markdown

DROP_DIRS = {"src"}
MERGE = {"practice": "prep", "quick-revision": "prep"}
MIN_CHARS = 150

# FLAN-style instruction prompt: the task, the closed label set spelled out,
# then the note. Instruction-tuned models need the label vocabulary in context
# to zero-shot; a bare "classify:" prefix makes them echo input fragments.
LABELS = [
    "backend",
    "behavioral",
    "dsa",
    "java",
    "learnings",
    "lld",
    "prep",
    "system-design",
]
PROMPT_TEMPLATE = (
    "Classify the interview note below into exactly one of these topics: "
    + ", ".join(LABELS)
    + ".\n\nNote: {text}\n\nTopic:"
)


def label_for(rel_path: str) -> str | None:
    parts = rel_path.split("/")
    if len(parts) == 1:
        return None  # root-level files are repo meta docs, not interview content
    top = parts[0]
    if top in DROP_DIRS:
        return None
    return MERGE.get(top, top)


def build_dataset(
    corpus: Path,
    max_chars: int = 400,
    overlap: int = 80,
    eval_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    # file -> list of (text, section)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for md in sorted(corpus.rglob("*.md")):
        rel = md.relative_to(corpus).as_posix()
        label = label_for(rel)
        if label is None:
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        for ch in chunk_markdown(text, rel, max_chars=max_chars, overlap=overlap):
            if len(ch.text) < MIN_CHARS:
                continue
            by_file[rel].append(
                {"text": ch.text, "section": ch.section, "file": rel, "label": label}
            )

    # Stratified FILE-level split: for each label, shuffle its files and take
    # eval_frac (at least one file) for eval.
    files_by_label: dict[str, list[str]] = defaultdict(list)
    for rel, items in by_file.items():
        files_by_label[items[0]["label"]].append(rel)

    train_files: set[str] = set()
    eval_files: set[str] = set()
    for label, files in sorted(files_by_label.items()):
        files = sorted(files)
        rng.shuffle(files)
        n_eval = max(1, round(len(files) * eval_frac))
        eval_files.update(files[:n_eval])
        train_files.update(files[n_eval:])

    train, evl = [], []
    for rel, items in by_file.items():
        (evl if rel in eval_files else train).extend(items)

    rng.shuffle(train)
    rng.shuffle(evl)

    labels = sorted({it["label"] for it in train} | {it["label"] for it in evl})
    label_ids = {lab: i for i, lab in enumerate(labels)}
    for it in train + evl:
        it["label_id"] = label_ids[it["label"]]
        it["input"] = PROMPT_TEMPLATE.format(text=it["text"])
        it["target"] = it["label"]

    stats = {
        "n_train": len(train),
        "n_eval": len(evl),
        "labels": labels,
        "train_dist": dict(Counter(it["label"] for it in train)),
        "eval_dist": dict(Counter(it["label"] for it in evl)),
        "train_files": sorted(train_files),
        "eval_files": sorted(eval_files),
        "max_chars": max_chars,
        "overlap": overlap,
        "eval_frac": eval_frac,
        "seed": seed,
        "min_chars": MIN_CHARS,
    }
    return train, evl, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, evl, stats = build_dataset(
        args.corpus,
        max_chars=args.max_chars,
        overlap=args.overlap,
        eval_frac=args.eval_frac,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n"
    )
    (args.out / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in evl) + "\n"
    )
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"train={len(train)} eval={len(evl)} labels={stats['labels']}")
    print("train dist:", stats["train_dist"])
    print("eval  dist:", stats["eval_dist"])
    print("eval files:", stats["eval_files"])


if __name__ == "__main__":
    main()
