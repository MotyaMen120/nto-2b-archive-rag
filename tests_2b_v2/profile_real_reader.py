"""Timing on two real questions per fixed validation fold. No labels enter inference."""
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import solution_2b_v2 as s


def main():
    root = s.ROOT
    train = s.read_questions(s.find_file(root, "train.csv"))
    folds, splits = s.group_folds(train, s.find_file(root, "train_groups.csv"))
    selected = np.concatenate([va[:2] for _, va in splits])
    questions = train.iloc[selected].reset_index(drop=True)
    output = root / "output_2b_final" / "reader_profile"
    output.mkdir(parents=True, exist_ok=True)
    cfg = s.Config(cache=output / "cache", lexical_only=True, encoder="", reader="Qwen/Qwen2.5-7B-Instruct",
                   input_tokens=4096, output_tokens=512, reader_batch=2)
    monitor = s.ResourceMonitor()
    retriever = s.Retriever(pd.read_parquet(s.find_file(root, "corpus.parquet")), cfg)
    items = s.prepare(questions, retriever)
    started = time.perf_counter()
    records = s.infer(questions, items, retriever, output / "raw.jsonl", s.pipeline_signature(cfg), folds[selected])
    elapsed = time.perf_counter() - started
    report = {"timing_only": True, "n": len(records), "folds": list(map(int, folds[selected])),
              "reader_seconds_including_load": elapsed, "seconds_per_question_including_load": elapsed / len(records),
              "parse_failures": sum(bool(r["parse_error"]) for r in records), "resources": monitor.stop(),
              "note": "Fixed reader and lexical retrieval, no training or metric tuning. This is not a full OOF score."}
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
