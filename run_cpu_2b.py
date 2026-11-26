"""Train the grouped CPU reader and write official OOF/test submissions, or predict with saved models."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import pyarrow.parquet as pq

import cpu_categorical_2b as categorical
import cpu_numeric_2b as numeric
import solution_2b_v2 as base


PUBLIC = ("qid", "question", "asked_at", "answer_format", "unit", "options")


def load_corpus(path):
    posts = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for post in batch.to_pylist():
            pid = int(post["post_id"])
            if pid in posts:
                raise ValueError(f"Duplicate post {pid}")
            post["date"] = str(post["date"])[:10]
            posts[pid] = post
    return posts


def ranked_rows(path, questions, corpus, limit=60):
    contexts = {}
    expected = {r["qid"]: r["asked_at"] for r in questions}
    with path.open(encoding="utf8") as stream:
        for line in stream:
            item = json.loads(line)
            qid = item["qid"]
            if qid in contexts or qid not in expected:
                raise ValueError(f"Unexpected or duplicate qid {qid} in {path}")
            ids = list(map(int, item["ids"]))
            if len(set(ids)) != len(ids) or len(ids) != len(item["scores"]):
                raise ValueError(f"Malformed ranking {qid}")
            if any(corpus[p]["date"] > expected[qid] for p in ids):
                raise ValueError(f"Evidence later than asked_at for {qid}")
            contexts[qid] = [dict(corpus[p], rank_score=s) for p, s in zip(ids[:limit], item["scores"])]
    if set(contexts) != set(expected):
        raise ValueError(f"Partial rankings in {path}")
    return contexts


def fit(rows, corpus, seeds):
    seed_ids = {row['qid']: {'ids': [p['post_id'] for p in seeds[row['qid']]]} for row in rows}
    return {
        "categorical": categorical.fit(rows, corpus, ranked=seeds),
        "numeric": numeric.fit_ranker(rows, corpus, seed_ids),
    }


def predict(models, rows, ranks):
    predictions = []
    for row in rows:
        question = {k: row.get(k, "") for k in PUBLIC}
        posts = ranks[question["qid"]]
        if question["answer_format"] in ("date", "number"):
            answer = numeric.predict(question, posts, models["numeric"])
        else:
            answer = categorical.predict(models["categorical"], question, posts)
        ids = [] if answer == "no_info" else [str(p["post_id"]) for p in posts[:5]]
        predictions.append({"qid": question["qid"], "answer": answer, "evidence": " ".join(ids)})
    return predictions


def write_checked(path, predictions, question_path, corpus_path):
    validator = base.find_file(base.ROOT, "submission.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf8", suffix=".csv",
                                     dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=["qid", "answer", "evidence"])
        writer.writeheader()
        writer.writerows(predictions)
    try:
        subprocess.run([sys.executable, str(validator), str(tmp), "--questions", str(question_path),
                        "--corpus", str(corpus_path)], check=True)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def evaluate(path, train_path, train, folds, ranks):
    scorer = base.module_from_file(base.find_file(base.ROOT, "score.py"))
    rows, report = scorer.run(str(path), [str(train_path)])
    gold = {x["id"]: x for x in scorer.load_gold([str(train_path)])}
    report["fold_S"] = [scorer.total([r for i, r in enumerate(rows) if int(folds[i]) == f]) for f in range(5)]
    report["by_format"] = {}
    report["by_level_format"] = {}
    for field, keys in (("by_format", ["answer_format"]), ("by_level_format", ["level", "answer_format"])):
        groups = {}
        for result in rows:
            key = " / ".join(gold[result["qid"]][k] for k in keys)
            groups.setdefault(key, []).append(result)
        for key, group in groups.items():
            report[field][key] = {"n": len(group), "S": scorer.total(group),
                                  "correct": sum(x["a"] > 0 for x in group),
                                  "mean_a": sum(x["a"] for x in group) / len(group),
                                  "mean_e": sum(x["e"] for x in group) / len(group)}
    oracle = []
    counts = Counter()
    rank_depth = max((len(posts) for posts in ranks.values()), default=0)
    for result in rows:
        g = gold[result["qid"]]
        ids = [p["post_id"] for p in ranks[g["id"]][:5]] if g["_G"] else []
        oracle.append(scorer.score_item(g, {"answer": g["answer"], "evidence": ids}))
        if result["a"] > 0:
            counts["correct"] += 1
        elif not g["_G"]:
            counts["missed_no_info"] += 1
        elif not {p["post_id"] for p in ranks[g["id"]]} & set(g["_G"]):
            counts[f"wrong_no_gold_in_top{rank_depth}"] += 1
        elif not set(ids) & set(g["_G"]):
            counts[f"wrong_gold_in_top{rank_depth}_outside_top5"] += 1
        else:
            counts["wrong_with_gold_in_top5"] += 1
    report["error_counts"] = counts
    report["S_if_answers_perfect_with_these_top5"] = scorer.total(oracle)
    report["correct"] = sum(r["a"] > 0 for r in rows)
    by_qid = {r["qid"]: r["question"] for r in train}
    report["largest_errors"] = [dict(r, question=by_qid[r["qid"]]) for r in
        sorted(rows, key=lambda r: -r["w"]*(1-r["s"]))[:15]]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "test", "corpus", "groups"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--rank-dir", type=Path, default=base.ROOT/"output_2b_final/cpu_rank")
    parser.add_argument("--seed-dir", type=Path, help="Fixed, unsupervised rankings for reader training")
    parser.add_argument("--nested-contexts", type=Path,
                        help="Use inner group-held-out rankings for each outer reader training fold")
    parser.add_argument("--output", type=Path, default=base.ROOT/"output_2b_final/cpu/submission.csv")
    parser.add_argument("--model", type=Path, help="Use saved reader models and predict test only")
    args = parser.parse_args()
    monitor = base.ResourceMonitor()
    corpus_path = base.find_file(base.ROOT, "corpus.parquet", args.corpus)
    test_path = base.find_file(base.ROOT, "test.csv", args.test)
    test = base.read_questions(test_path).to_dict("records")
    corpus = load_corpus(corpus_path)
    report = {}
    destination = args.output.parent
    destination.mkdir(parents=True, exist_ok=True)
    if args.model:
        models = joblib.load(args.model)
    else:
        train_path = base.find_file(base.ROOT, "train.csv", args.train)
        train_frame = base.read_questions(train_path)
        groups_path = base.find_file(base.ROOT, "train_groups.csv", args.groups)
        folds, splits = base.group_folds(train_frame, groups_path)
        train = train_frame.to_dict("records")
        rankings = ranked_rows(args.rank_dir/"train.jsonl", train, corpus)
        if args.nested_contexts:
            signature = json.loads((args.nested_contexts/"inputs.json").read_text(encoding="utf8"))
            for key, path in {"train": train_path, "groups": groups_path, "corpus": corpus_path,
                              "outer_rank": args.rank_dir/"train.jsonl"}.items():
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != signature[key]:
                        raise ValueError(f"Nested training contexts differ for {key}")
        else:
            seeds = ranked_rows((args.seed_dir or args.rank_dir)/"seed_train.jsonl", train, corpus)
        preds = {}
        for fold, (tr, va) in enumerate(splits, 1):
            fit_rows = [train[i] for i in tr]
            if args.nested_contexts:
                seeds = ranked_rows(args.nested_contexts/f"fold_{fold-1}.jsonl", fit_rows, corpus)
            models = fit(fit_rows, corpus, seeds)
            for pred in predict(models, [train[i] for i in va], rankings):
                preds[pred["qid"]] = pred
            print(f"Reader fold {fold}/5 complete", flush=True)
        oof = destination/"oof.csv"
        write_checked(oof, [preds[r["qid"]] for r in train], train_path, corpus_path)
        report = evaluate(oof, train_path, train, folds, rankings)
        print(f"Official grouped OOF S = {report['S']:.4f}", flush=True)
        models = fit(train, corpus, rankings if args.nested_contexts else seeds)
        joblib.dump(models, destination/"readers.joblib", compress=3)
    ranks = ranked_rows(args.rank_dir/"test.jsonl", test, corpus)
    predicted = predict(models, test, ranks)
    write_checked(args.output, predicted, test_path, corpus_path)
    report["reader_resources"] = monitor.stop()
    report["test"] = {"n": len(predicted), "formats": dict(Counter(r["answer_format"] for r in test)),
                      "validated": True, "future_evidence": 0}
    (destination/("test.report.json" if args.model else "report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
