"""LoRA fine-tune of a seq2seq model for handbook topic classification.

Task framing (classification as generation, FLAN-style instruction prompt):
  input : "Classify the interview note below into exactly one of these
           topics: backend, behavioral, dsa, java, learnings, lld, prep,
           system-design.\n\nNote: <chunk text>\n\nTopic:"
  target: "<label>"                      e.g. "system-design"

Default model: google/flan-t5-base (250M). LoRA keeps trainable params tiny
(~0.5% of weights), so the full run fits on a free Colab T4 in minutes.

Smoke mode (--smoke) runs a handful of steps on CPU with flan-t5-small to
validate the loop without a GPU; it is what CI exercises.

Usage (Colab GPU):
  python train.py --train data/train.jsonl --out checkpoints/lora \
      --model google/flan-t5-base --epochs 5 --batch-size 8 --lr 3e-4

Smoke (CPU, no GPU needed):
  python train.py --train data/train.jsonl --out /tmp/smoke --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Sanely-chosen defaults, documented for the README:
#  - rank 16 / alpha 32: standard LoRA starting point for small models; rank 8
#    underfits multi-class topics, rank 64 adds params with no observed gain
#    on <500 examples (would need a GPU run to verify — stated as untested).
#  - q,v only: the two attention projections carry most adaptation signal per
#    the original LoRA paper; keeps trainable params ~0.4% of flan-t5-base.
#  - lr 3e-4: standard for LoRA on T5-family models.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q", "v"]
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 16


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_hf_dataset(rows: list[dict], tokenizer, max_steps_subset: int = 0):
    if max_steps_subset:
        rows = rows[:max_steps_subset]

    def tok(row):
        model_in = tokenizer(
            row["input"], truncation=True, max_length=MAX_INPUT_LEN
        )
        # transformers>=5 removed as_target_tokenizer; T5 uses one
        # tokenizer for inputs and targets, so plain tokenize is correct.
        labels = tokenizer(
            row["target"], truncation=True, max_length=MAX_TARGET_LEN
        )
        model_in["labels"] = labels["input_ids"]
        return model_in

    # Drop every source column: the dataset keeps only input_ids /
    # attention_mask / labels. (Needed because the raw rows contain a
    # string column named "label", which DataCollatorForSeq2Seq would
    # otherwise mistake for the token-id labels.)
    return Dataset.from_list(rows).map(
        tok, remove_columns=list(rows[0].keys())
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU validation: flan-t5-small, 16 examples, few steps")
    args = ap.parse_args()

    model_name = "google/flan-t5-small" if args.smoke else args.model
    rows = load_jsonl(args.train)
    print(f"model={model_name} train_rows={len(rows)} smoke={args.smoke}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    ds = build_hf_dataset(rows, tokenizer,
                          max_steps_subset=16 if args.smoke else 0)

    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=2 if args.smoke else args.batch_size,
        num_train_epochs=1 if args.smoke else args.epochs,
        max_steps=4 if args.smoke else -1,
        learning_rate=args.lr,
        seed=args.seed,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=bool(args.smoke),
        fp16=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    if not args.smoke:
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        print(f"adapter saved to {args.out}")
    else:
        print("smoke run OK: training loop validated (no checkpoint saved)")


if __name__ == "__main__":
    main()
