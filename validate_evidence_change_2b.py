"""Measure a citation-only change against the submitted reader, with paired story bootstrap."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import solution_2b_v2 as base
from predict_offline_2b import check_dates, digest
from run_cpu_2b import write_checked


def replace_evidence(rows, source, rankings):
    if set(source) != set(rankings) or set(source) != {q["qid"] for q in rows}:
        raise ValueError("Answer and evidence files must cover exactly the same question IDs")
    result = []
    for q in rows:
        qid = q["qid"]
        old = source[qid]
        ids = rankings[qid]["ids"][:5]
        result.append(dict(old, evidence="" if old["answer"] == "no_info" else " ".join(map(str, ids))))
    return result


def compare(scorer, gold, old, new, groups, folds):
    pairs = []
    for g, fold in zip(gold, folds):
        qid = g["id"]
        scored = [scorer.score_item(g, dict(answer=p[qid]["answer"], evidence=p[qid]["evidence"].split()))
                  for p in (old, new)]
        pairs.append((groups[qid], int(fold), *scored))
    before = [a for _,_,a,_ in pairs]
    after = [b for _,_,_,b in pairs]
    group_data = {}
    for group, _, a, b in pairs:
        group_data.setdefault(group, np.zeros(3))
        group_data[group] += [a["w"]*a["s"], b["w"]*b["s"], a["w"]]
    data = np.array(list(group_data.values()))
    rng = np.random.default_rng(20261125)
    draws = data[rng.integers(len(data), size=(5000, len(data)))].sum(axis=1)
    deltas = 100*(np.maximum(0,draws[:,1])-np.maximum(0,draws[:,0]))/draws[:,2]
    scores = [scorer.total(before), scorer.total(after)]
    return {"n": len(pairs), "groups": len(data), "old_S": scores[0], "new_S": scores[1],
            "delta_S": scores[1]-scores[0], "delta_ci95": np.quantile(deltas,[.025,.975]).tolist(),
            "changed_answers": sum(a["answer"] != b["answer"] for _,_,a,b in pairs),
            "fold_pairs": [[scorer.total([p[k] for p in pairs if p[1]==i]) for k in (2,3)] for i in range(5)],
            "levels": {level: [scorer.total([p[k] for p in pairs if p[k]["level"]==level]) for k in (2,3)]
                       for level in ("light","medium","heavy")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "test", "groups", "corpus"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--reader-dir", type=Path, required=True)
    parser.add_argument("--rank-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = base.find_file(base.ROOT, "corpus.parquet", args.corpus)
    groups_file = base.find_file(base.ROOT, "train_groups.csv", args.groups)
    scorer = base.module_from_file(base.find_file(base.ROOT, "score.py"))
    groups = pd.read_csv(groups_file,dtype=str).set_index("qid").group.to_dict()
    report = {}
    for split, filename in (("train", "oof.csv"), ("test", "submission.csv")):
        path = base.find_file(base.ROOT, split+".csv", getattr(args, split))
        frame = base.read_questions(path)
        rankpath = args.rank_dir/(split+".jsonl")
        ranks = {r["qid"]:r for r in map(json.loads, rankpath.read_text(encoding="utf8").splitlines())}
        with (args.reader_dir/filename).open(encoding="utf8", newline="") as f:
            before = {r["qid"]:r for r in csv.DictReader(f)}
        predicted = replace_evidence(frame.to_dict("records"), before, ranks)
        dest = args.output if split == "test" else args.output.with_name("oof.csv")
        write_checked(dest, predicted, path, corpus)
        check_dates(dest, path, corpus)
        report[split] = {"sha256":digest(dest), "questions":len(predicted),
                         "changed_citations":sum(before[r["qid"]]["evidence"]!=r["evidence"] for r in predicted)}
        if split == "train":
            folds,_ = base.group_folds(frame, groups_file)
            report["paired"] = compare(scorer, scorer.load_gold([str(path)]), before,
                                       {r["qid"]:r for r in predicted}, groups, folds)
    args.output.with_name("report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
