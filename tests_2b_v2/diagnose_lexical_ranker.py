"""Real-data GroupKFold evidence-only baseline, using the official AP@5 function."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import solution_2b_v2 as s
import meta_2b as meta


def main():
    root = s.ROOT
    output = root / "output_2b_final" / "lexical_ranker_baseline"
    output.mkdir(parents=True, exist_ok=True)
    train_path = s.find_file(root, "train.csv")
    questions = s.read_questions(train_path)
    folds, splits = s.group_folds(questions, s.find_file(root, "train_groups.csv"))
    cfg = s.Config(cache=output / "cache", lexical_only=True, encoder="", reader="")
    retriever = s.Retriever(pd.read_parquet(s.find_file(root, "corpus.parquet")), cfg)
    items = s.prepare(questions, retriever)
    scorer = s.module_from_file(s.find_file(root, "score.py"))
    gold = scorer.load_gold([str(train_path)])
    gold_by_id = {g["id"]: g for g in gold}
    records = [{"qid": item["qid"], "meta_candidates": item["meta_candidates"], "meta_features": item["meta_features"]} for item in items]
    predicted = [None] * len(records)
    for fold, (tr, va) in enumerate(splits):
        model = meta.fit_ranker(records, tr, gold_by_id)
        for i in va:
            ids = records[i]["meta_candidates"]
            scores = model.booster_.predict(records[i]["meta_features"]) if ids else []
            predicted[i] = [ids[j] for j in sorted(range(len(ids)), key=lambda j: (-scores[j], ids[j]))]
        print(f"Lexical evidence OOF {fold + 1}/5", flush=True)
    rows = []
    for i, question in enumerate(questions.to_dict("records")):
        item = gold_by_id[question["qid"]]
        gold_ids = item["_G"]
        for pid in gold_ids:
            if pid not in retriever.by_id or retriever.dates[retriever.by_id[pid]] > np.datetime64(question["asked_at"], "D"):
                raise ValueError(f"Gold evidence temporal integrity failure for {question['qid']}: {pid}")
        if not gold_ids:
            continue
        raw = records[i]["meta_candidates"]
        rows.append({"qid": question["qid"], "level": question["level"], "answer_format": question["answer_format"],
                     "fold": int(folds[i]), "w": item["_w"], "baseline_ap5": scorer.ap5(raw[:5], gold_ids),
                     "meta_ap5": scorer.ap5(predicted[i][:5], gold_ids)})
    def summarize(subset):
        weights = np.asarray([r["w"] for r in subset])
        return {"n": len(subset), "baseline_ap5": float(np.mean([r["baseline_ap5"] for r in subset])),
                "meta_ap5": float(np.mean([r["meta_ap5"] for r in subset])),
                "baseline_weighted_ap5": float(np.average([r["baseline_ap5"] for r in subset], weights=weights)),
                "meta_weighted_ap5": float(np.average([r["meta_ap5"] for r in subset], weights=weights))}
    report = {"evidence_only": True, "official_S": None, "n_questions": len(questions),
              "note": "Fixed lexical retrieval; LightGBM trained strictly on other groups. Empty-gold no_info excluded from evidence-only means.",
              "all": summarize(rows),
              "by_level": {level: summarize([r for r in rows if r["level"] == level]) for level in ["light", "medium", "heavy"]},
              "by_fold": {str(f): summarize([r for r in rows if r["fold"] == f]) for f in range(5)}}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
