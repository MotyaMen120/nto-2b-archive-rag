"""Estimate the official score gain over skip for numerical and date answers."""
from collections import Counter
import math
import re

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cpu_numeric_2b as number


def confidence_features(row, posts):
    question = row["question"].lower()
    dates = [p["date"] for p in posts[:10]]
    chan = [p["channel"] for p in posts[:5]]
    stem = lambda s: {m[:5] for m in re.findall(r"[а-яё]{4,}", s.lower())}
    q = stem(question)
    cov = [len(q & stem(str(p["text"]))) / max(1,len(q)) for p in posts[:5]]
    is_delta = bool(re.search(r"между|на сколько|сколько (?:полных )?(?:сут|дней|месяц|раз)|разниц",question))
    fmt = row["answer_format"]
    unit = row.get("unit","")
    return {"fmt_"+fmt:1.,"unit_"+unit:1.,"length":math.log1p(len(q)), "delta":float(is_delta),
            "first_news":float("впервые" in question),"comparison":float(bool(re.search(r"больш|меньш|дешев|дороже",question))),
            "n_dates":len(set(dates))/10., "same_day":max(Counter(dates).values(),default=0)/10.,
            "channels":len(set(chan))/5., "coverage_best":max(cov,default=0.),"coverage_avg":sum(cov)/5.,
            "days":float(unit in ("дней","месяцев")), "qn":len(re.findall(r"\d+",question))/5.}


def utility_row(row,posts,scorer,gold_item):
    answer, confidence = number.predict_with_confidence(row,posts)
    ranked = [p["post_id"] for p in posts[:5]] if answer!="no_info" else []
    utility = scorer.score_item(gold_item,{"answer":answer,"evidence":ranked})["s"] + .175
    features = confidence_features(row,posts)
    features["confidence"] = float(confidence)
    return features,utility


def prepare(rows,seeds,scorer,gold):
    return {r["qid"]:utility_row(r,seeds[r["qid"]],scorer,gold[r["qid"]])
            for r in rows if r["answer_format"] in ("number","date")}


def fit(rows, prepared, gold):
    qids = [r["qid"] for r in rows if r["qid"] in prepared]
    if len(qids)<20:
        return None
    # Standardize only numeric features. Sparse, uncommon units are regularized.
    reg = make_pipeline(DictVectorizer(sparse=False),StandardScaler(),Ridge(alpha=100))
    xs = [prepared[k][0] for k in qids]
    ys = [prepared[k][1] for k in qids]
    weights = np.asarray([gold[k]["_w"] for k in qids],dtype=float)
    weights /= weights.mean()
    reg.fit(xs,ys,ridge__sample_weight=weights)
    return reg


def prefer_skip(model,row,seed_posts,answer):
    if model is None or answer in ("skip","no_info","promised"):
        return False
    features = confidence_features(row,seed_posts)
    _, confidence = number.predict_with_confidence(row,seed_posts)
    features["confidence"] = float(confidence)
    return model.predict([features])[0] < 0

def main():
    import argparse
    import csv
    import json
    from pathlib import Path
    import joblib
    import solution_2b_v2 as base
    from run_cpu_2b import evaluate, load_corpus, ranked_rows, write_checked
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("train","test","groups","corpus","model"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--input",type=Path,required=True,help="Directory containing the reader submission.csv and oof.csv")
    parser.add_argument("--rank-dir",type=Path,required=True)
    parser.add_argument("--seed-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    corpus_path=base.find_file(base.ROOT,"corpus.parquet",args.corpus)
    corpus=load_corpus(corpus_path)
    def table(path):
        with path.open(encoding="utf8",newline="") as stream:
            return {row["qid"]:row for row in csv.DictReader(stream)}
    def apply(model,rows,records,seed):
        for r in rows:
            if (r["answer_format"] in ("number","date") and
                    prefer_skip(model,r,seed[r["qid"]],records[r["qid"]]["answer"])):
                records[r["qid"]]["answer"]="skip"
    if args.model:
        model=joblib.load(args.model)
    else:
        train_path=base.find_file(base.ROOT,"train.csv",args.train)
        frame=base.read_questions(train_path)
        folds,splits=base.group_folds(frame,base.find_file(base.ROOT,"train_groups.csv",args.groups))
        rows=frame.to_dict("records")
        records=table(args.input/"oof.csv")
        seed=ranked_rows(args.seed_dir/"seed_train.jsonl",rows,corpus)
        scorer=base.module_from_file(base.find_file(base.ROOT,"score.py"))
        gold={r["id"]:r for r in scorer.load_gold([str(train_path)])}
        ready=prepare(rows,seed,scorer,gold)
        for tr,va in splits:
            model=fit([rows[i] for i in tr],ready,gold)
            apply(model,[rows[i] for i in va],records,seed)
        out=args.output.parent/"oof.csv"
        write_checked(out,[records[r["qid"]] for r in rows],train_path,corpus_path)
        ranks=ranked_rows(args.rank_dir/"train.jsonl",rows,corpus)
        report=evaluate(out,train_path,rows,folds,ranks)
        (args.output.parent/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
        print(f"Calibrated official OOF S = {report['S']:.4f}",flush=True)
        model=fit(rows,ready,gold)
        joblib.dump(model,args.output.parent/"policy.joblib")
    test_path=base.find_file(base.ROOT,"test.csv",args.test)
    test=base.read_questions(test_path).to_dict("records")
    seed=ranked_rows(args.seed_dir/"seed_test.jsonl",test,corpus)
    records=table(args.input/"submission.csv")
    apply(model,test,records,seed)
    write_checked(args.output,[records[r["qid"]] for r in test],test_path,corpus_path)


if __name__=="__main__":
    main()
