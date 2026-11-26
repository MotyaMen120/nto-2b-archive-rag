"""Optional real-model integration on synthetic data, without train labels or network."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import solution_2b_v2 as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=s.ROOT / "output_2b_v2" / "synthetic_integration")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitor = s.ResourceMonitor()
    posts = pd.DataFrame([
        {"post_id": 1001, "date": "2024-02-01", "channel": "Канал A", "channel_type": "vk", "text": "Парк запланировали открыть 15 марта 2024 года. Работы продолжаются."},
        {"post_id": 1002, "date": "2024-02-03", "channel": "Канал B", "channel_type": "tg", "text": "Парк сегодня открыли для посетителей. Работы завершены."},
    ])
    questions = pd.DataFrame([
        {"qid": "synthetic-status", "question": "Какой статус открытия парка?", "asked_at": "2024-02-01", "answer_format": "status", "unit": "", "options": ""},
        {"qid": "synthetic-math", "question": "Сколько дней прошло от публикации анонса открытия парка до публикации новости о его открытии?", "asked_at": "2024-02-03", "answer_format": "number", "unit": "дней", "options": ""},
    ])
    cfg = s.Config(cache=args.output_dir / "cache", reader=args.reader, lexical_only=True, verify="none", context_posts=4, input_tokens=4096)
    retriever = s.Retriever(posts, cfg)
    items = s.prepare(questions, retriever)
    records = s.infer(questions, items, retriever, args.output_dir / "raw.jsonl", s.pipeline_signature(cfg))
    predictions = {r["qid"]: r["answer"] for r in records}
    expected = {"synthetic-status": "promised", "synthetic-math": "2"}
    if predictions != expected:
        raise AssertionError(f"Synthetic integration mismatch: {predictions}; inspect raw.jsonl.")
    questions_path, corpus_path = args.output_dir / "questions.csv", args.output_dir / "corpus.parquet"
    questions.to_csv(questions_path, index=False)
    posts.to_parquet(corpus_path, index=False)
    s.write_submission(records, questions, retriever, args.output_dir / "submission.csv", s.find_file(s.ROOT, "submission.py"), corpus_path, questions_path)
    result = {"synthetic_only": True, "expected": expected, "actual": predictions, "resources": monitor.stop()}
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
