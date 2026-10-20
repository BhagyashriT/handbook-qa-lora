"""Unit tests for the dataset pipeline (stdlib only — no torch needed)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_prep import PROMPT_TEMPLATE, build_dataset, label_for


def write_corpus(root: Path) -> None:
    # Two labels x two files each; content long enough to survive MIN_CHARS.
    body = ("Some interview content about the topic. " * 12).strip()
    (root / "java").mkdir()
    (root / "dsa").mkdir()
    (root / "practice").mkdir()
    (root / "src").mkdir()
    (root / "java" / "a.md").write_text(f"# Java A\n\n## Sec\n\n{body}\n")
    (root / "java" / "b.md").write_text(f"# Java B\n\n## Sec\n\n{body}\n")
    (root / "dsa" / "a.md").write_text(f"# DSA A\n\n## Sec\n\n{body}\n")
    (root / "dsa" / "b.md").write_text(f"# DSA B\n\n## Sec\n\n{body}\n")
    (root / "practice" / "q.md").write_text(f"# Q\n\n## Sec\n\n{body}\n")
    (root / "practice" / "p.md").write_text(f"# P\n\n## Sec\n\n{body}\n")
    (root / "src" / "README.md").write_text("# src stub\n")
    (root / "README.md").write_text("# repo readme\n")


class TestLabels(unittest.TestCase):
    def test_directory_labels(self):
        self.assertEqual(label_for("java/concurrency.md"), "java")
        self.assertEqual(label_for("system-design/chat.md"), "system-design")

    def test_prep_merge(self):
        self.assertEqual(label_for("practice/question-bank.md"), "prep")
        self.assertEqual(label_for("quick-revision/30-minute.md"), "prep")

    def test_dropped(self):
        self.assertIsNone(label_for("src/README.md"))
        self.assertIsNone(label_for("README.md"))
        self.assertIsNone(label_for("RESOURCES.md"))


class TestDataset(unittest.TestCase):
    def test_split_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_corpus(root)
            train, evl, stats = build_dataset(root, seed=7)

            # No file appears in both splits (leakage guard).
            train_files = {r["file"] for r in train}
            eval_files = {r["file"] for r in evl}
            self.assertTrue(train_files.isdisjoint(eval_files))

            # Dropped content never appears.
            for r in train + evl:
                self.assertNotIn("src/", r["file"])
                self.assertNotEqual(r["file"], "README.md")

            # Merge applied.
            self.assertIn("prep", stats["labels"])
            self.assertNotIn("practice", stats["labels"])

            # Prompt/target contract for the seq2seq task.
            r0 = train[0]
            self.assertTrue(r0["input"].startswith("Classify the interview note"))
            self.assertTrue(r0["input"].endswith("Topic:"))
            self.assertIn(r0["text"], r0["input"])
            self.assertEqual(r0["target"], r0["label"])
            self.assertEqual(r0["label_id"], stats["labels"].index(r0["label"]))

            # Every label with >=2 files lands in both splits (stratified).
            for lab in ("java", "dsa"):
                self.assertIn(lab, stats["train_dist"])
                self.assertIn(lab, stats["eval_dist"])

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_corpus(root)
            t1, e1, _ = build_dataset(root, seed=123)
            t2, e2, _ = build_dataset(root, seed=123)
            self.assertEqual([r["file"] for r in t1], [r["file"] for r in t2])
            self.assertEqual([r["file"] for r in e1], [r["file"] for r in e2])

    def test_stats_json_serializable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_corpus(root)
            _, _, stats = build_dataset(root, seed=7)
            json.dumps(stats)  # must not raise


if __name__ == "__main__":
    unittest.main()
