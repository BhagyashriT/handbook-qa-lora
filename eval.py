"""Evaluate base model vs LoRA adapter on the held-out split.

For each eval example, generates the label with greedy decoding and compares
to the gold label (exact match after strip/lower). Reports accuracy and
per-class F1 (sklearn), plus a base-vs-adapter comparison table.

Usage:
  # zero-shot base model (CPU-ok for flan-t5-base on the eval split)
  python eval.py --eval data/eval.jsonl --model google/flan-t5-base

  # after training on Colab: compare the adapter against the base
  python eval.py --eval data/eval.jsonl --model google/flan-t5-base \
      --adapter checkpoints/lora

  # quick documented subset (e.g. slow CPU)
  python eval.py --eval data/eval.jsonl --limit 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_INPUT_LEN = 512


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(s: str) -> str:
    return s.strip().lower()


@torch.inference_mode()
def predict(rows: list[dict], model, tokenizer, device: str) -> list[str]:
    preds = []
    for row in rows:
        ids = tokenizer(
            row["input"], return_tensors="pt",
            truncation=True, max_length=MAX_INPUT_LEN,
        ).input_ids.to(device)
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
        preds.append(normalize(tokenizer.decode(out[0], skip_special_tokens=True)))
    return preds


def report(name: str, gold: list[str], preds: list[str], labels: list[str]) -> dict:
    acc = accuracy_score(gold, preds)
    f1s = f1_score(gold, preds, labels=labels, average=None, zero_division=0)
    macro = f1_score(gold, preds, labels=labels, average="macro", zero_division=0)
    print(f"\n== {name} ==")
    print(f"accuracy: {acc:.3f}   macro-F1: {macro:.3f}   n={len(gold)}")
    print(f"{'label':16s} {'f1':>6s} {'n':>4s}")
    for lab, f1 in zip(labels, f1s):
        n = sum(1 for g in gold if g == lab)
        print(f"{lab:16s} {f1:6.3f} {n:4d}")
    return {"accuracy": acc, "macro_f1": macro,
            "per_class_f1": dict(zip(labels, (round(float(f), 3) for f in f1s)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--adapter", default=None,
                    help="path to LoRA adapter checkpoint; also evaluates base")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate on first N examples only (document when used)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write JSON results here")
    args = ap.parse_args()

    rows = load_jsonl(args.eval)
    if args.limit:
        rows = rows[: args.limit]
        print(f"NOTE: evaluating on a documented subset of {len(rows)} examples")
    gold = [normalize(r["target"]) for r in rows]
    labels = sorted(set(gold))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} n={len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    results = {"base": report("base (zero-shot)", gold,
                              predict(rows, base, tokenizer, device), labels)}
    if args.adapter:
        tuned = PeftModel.from_pretrained(base, args.adapter).eval()
        results["lora"] = report(f"LoRA adapter ({args.adapter})", gold,
                                 predict(rows, tuned, tokenizer, device), labels)
        b, l = results["base"]["accuracy"], results["lora"]["accuracy"]
        print(f"\naccuracy delta (lora - base): {l - b:+.3f}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
