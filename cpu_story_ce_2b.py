"""Offline larger multilingual cross-encoder over a fixed, label-free evidence shortlist."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import cpu_numeric_2b as numeric
import cpu_rank_2b as lexical
import cpu_semantic_2b as semantic
import cpu_story_2b as story
import solution_2b_v2 as base


def ordered(item, key="scores"):
    return [int(item["ids"][j]) for j in np.argsort(-np.array(item[key]),kind="stable")]


def shortlist(seed, small, dense, e5):
    lists=[(ordered(small),2.3), (ordered(e5),1.), (ordered(dense),.65),
           (ordered(seed),.7), (ordered(e5,"feedback_scores"),.7)]
    for part in e5.get("hop_top",[]):
        lists.append((part,.6))
    fusion={}
    for order,weight in lists:
        for i,pid in enumerate(order[:50]):
            fusion[pid]=fusion.get(pid,0.)+weight/(9+i)
    # Preserve independent first-stage candidates even when the two encoders disagree.
    mandatory=lists[0][0][:12]+lists[1][0][:3]
    return list(dict.fromkeys(mandatory+sorted(fusion,key=lambda p:(-fusion[p],p))))


def evidence_queries(row):
    import cpu_categorical_2b as categorical
    import cpu_nli_2b as nli
    import cpu_timeline_2b as timeline

    if timeline.supported(row):
        left, right = timeline.clauses(numeric._norm(row["question"]))
        return [(left, 0), (right, 1)]
    if row["answer_format"] == "choice":
        return [(nli.choice_claim(row["question"], opt), None)
                for opt in categorical.read_options(row.get("options"))]
    return [(row["question"], None)]


def facet_scores(frame, sources, scores, posts, engine, args, version):
    """Score question facets on date-eligible, label-free shortlists for each question."""
    cache = args.output / "facets"
    cache.mkdir(exist_ok=True)
    for row in frame.to_dict("records"):
        qid = row["qid"]
        item = scores[qid]
        pool = set(item["ids"])
        main_order = ordered(item)
        extra = []
        queries = evidence_queries(row)
        for query, role in queries:
            preference = []
            if role is not None:
                for source in (sources[3], sources[2]):
                    hops = source[qid].get("hop_top", [])
                    if len(hops) > role:
                        preference.extend(pid for pid in hops[role][:6] if pid in pool)
            best = list(dict.fromkeys(preference + main_order))
            ids = best[:20 if len(queries) <= 2 else 10]
            extra.append((query, ids))
        key = hashlib.sha256(json.dumps([version, qid, row["asked_at"], extra],
                                       ensure_ascii=False).encode()).hexdigest()
        filename = cache / (key + ".json")
        if filename.exists():
            item["facets"] = json.loads(filename.read_text(encoding="utf8"))
            continue
        values = []
        for query, ids in extra:
            words = numeric._words(numeric._norm(query))
            if any(str(posts.at[pid, "date"])[:10] > row["asked_at"] for pid in ids):
                raise ValueError("Future post in facet shortlist")
            pairs = [(query, str(posts.at[pid, "date"])[:10] + ". " +
                      article_excerpt(str(posts.at[pid, "text"]), words)) for pid in ids]
            values.append({"ids": ids, "scores": engine.encode(pairs, 4, False).astype(float).tolist()})
        item["facets"] = values
        temp = filename.with_suffix(".tmp")
        temp.write_text(json.dumps(values), encoding="utf8")
        temp.replace(filename)


def article_excerpt(text, query_words):
    import re
    if len(text)<1000:
        return text
    s=[x for x in re.split(r"(?<=[.!?])\s+|\n+",text) if x.strip()]
    if len(s)<2:
        return text
    overlap=[len(query_words & numeric._words(x.lower().replace("ё","е")))/(4+len(x)/25)**.5 for x in s]
    pos=int(np.argmax(overlap))
    if pos < 2:
        return text
    return s[0][:220]+" "+s[pos]+" "+" ".join(s[max(1,pos-1):pos]+s[pos+1:pos+3])


def logits(split,frame,corpus,args,which):
    deterministic=[semantic.read_ranks(path) for path in (args.lex_dir/f"seed_{split}.jsonl",
                   args.semantic_dir/f"ce_{split}.jsonl",args.semantic_dir/f"dense_{split}.jsonl",args.semantic_dir/f"e5_{split}.jsonl")]
    s=semantic.Engine(args.large_weights,args.threads,224)
    version="jina-32-v1"+semantic.digest(args.corpus)+semantic.digest(args.large_weights/"model.onnx")+semantic.digest(args.large_weights/"tokenizer.json")
    cache=args.output/"pairs";cache.mkdir(exist_ok=True,parents=True)
    posts=corpus.set_index("post_id"); result={};started=time.monotonic();fresh=0
    for idx in which:
        row=frame.iloc[idx].to_dict(); qid=row["qid"]
        ids=shortlist(*(src[qid] for src in deterministic))[:args.depth]
        if any(str(posts.at[p,"date"])[:10]>row["asked_at"] for p in ids):
            raise ValueError("Future document in cross-encoder shortlist")
        query=base.rewrite(row)["canonical_query"]
        if row["answer_format"]=="choice" and row.get("options"):
            query+=" "+" ".join(json.loads(row["options"]))
        key=hashlib.sha256(json.dumps([version,query,row["asked_at"],ids],ensure_ascii=False).encode()).hexdigest()
        path=cache/f"{key}.json"
        if path.is_file():
            data=json.loads(path.read_text(encoding="utf8"))
        else:
            qwords=numeric._words(query.lower().replace("ё","е"))
            pairs=[(query,str(posts.at[p,"date"])[:10]+". "+article_excerpt(str(posts.at[p,"text"]),qwords)) for p in ids]
            # One question, one cutoff. Dynamic quantization never pools articles across queries.
            vals=s.encode(pairs,4,False)
            data={"qid":qid,"ids":ids,"scores":vals.astype(float).tolist()}
            tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(data),encoding="utf8");tmp.replace(path)
            fresh+=len(ids)
        result[qid]=data
        if len(result)%20==0:
            print(f"{split} {len(result)} / {len(which)}; {fresh} new pairs in {time.monotonic()-started:.1f}s",flush=True)
    if args.facets:
        facet_scores(frame.iloc[which], deterministic, result, posts, s, args, version)
    return result


def augmented(split, frame, engine, corpus_dates, param, idx, large):
    raw_ids,raw_x=semantic.inputs(split,frame.assign(_day=frame.asked_at.apply(date.fromisoformat)),param,
                                  corpus_dates,engine)
    for i in idx:
        row=frame.iloc[i].to_dict();r=large[row["qid"]]
        ids=raw_ids[i];table=dict(zip(r["ids"],r["scores"]))
        ordered_ids=sorted(table,key=table.get,reverse=True)
        positions={p:j for j,p in enumerate(ordered_ids)}
        block=np.full((len(ids),3),np.nan,np.float32)
        for j,p in enumerate(ids):
            if p in table:
                block[j]=[table[p],table[p]-table[ordered_ids[0]],1/(1+positions[p])]
        blocks=[raw_x[i],block,engine.features(row,ids,table)]
        if "facets" in r:
            signals=np.full((len(ids),7),np.nan,np.float32)
            positions={p:k for k,p in enumerate(ids)}
            by_pid={}
            for facet in r["facets"]:
                order=ordered(facet)
                lookup=dict(zip(facet["ids"],facet["scores"]))
                for k,p in enumerate(order):
                    by_pid.setdefault(p,[]).append((lookup[p],lookup[p]-lookup[order[0]],1/(k+1)))
            for p,vals in by_pid.items():
                if p not in positions:
                    continue
                vals=np.asarray(vals)
                signals[positions[p]]=[*vals.max(axis=0),*vals.min(axis=0),len(vals)]
            blocks.append(signals)
        raw_x[i]=np.column_stack(blocks)
    return raw_ids,raw_x


def nested_reader_contexts(frame, splits, groups_path, train_path, ids, features, gold, args):
    """Produce retrieval contexts for reader training without using either held-out fold."""
    from sklearn.model_selection import GroupKFold

    target = args.nested_reader_contexts
    target.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(groups_path, dtype=str).set_index("qid").group.loc[frame.qid].to_numpy()
    for outer, (train_idx, valid_idx) in enumerate(splits):
        outer_groups = set(groups[valid_idx])
        records = {}
        for inner_train, inner_valid in GroupKFold(5).split(train_idx, groups=groups[train_idx]):
            fit_idx = train_idx[inner_train]
            predict_idx = train_idx[inner_valid]
            if set(groups[fit_idx]) & (outer_groups | set(groups[predict_idx])):
                raise ValueError("Overlapping evidence ranker training and evaluation groups")
            model = lexical.ranker(features, ids, frame, gold, fit_idx, args.threads)
            for record in lexical.apply(model, frame, ids, features, predict_idx, args.threads):
                records[record["qid"]] = record
        expected = frame.iloc[train_idx].qid.tolist()
        if set(records) != set(expected):
            raise ValueError("Incomplete inner predictions")
        lexical.dump_jsonl(target / f"fold_{outer}.jsonl", (records[qid] for qid in expected))
        print(f"Nested retrieval contexts {outer+1}/5", flush=True)
    fingerprints = {key: semantic.digest(path) for key, path in {
        "train": train_path, "groups": groups_path, "corpus": args.corpus,
        "outer_rank": args.output / "train.jsonl",
    }.items()}
    (target / "inputs.json").write_text(json.dumps(fingerprints, indent=2), encoding="utf8")


def pilot_scores(frame,which,folds,ranks,large,gold,scorer):
    cases=[];tentative=[];byfold={}
    for i in which:
        q=frame.iloc[i].qid; g=gold[q]; b=ranks[q];j=large[q]
        if not g["_G"]: continue
        base_pos={p:1/(6+k) for k,p in enumerate(b["ids"])};order=ordered(j)
        new_pos={p:1/(6+k) for k,p in enumerate(order)}
        rescored=sorted(b["ids"],key=lambda p:-base_pos[p]-.7*new_pos.get(p,0.))
        x=scorer.ap5(b["ids"][:5],g["_G"]); y=scorer.ap5(rescored[:5],g["_G"])
        cases.append(x);tentative.append(y);byfold.setdefault(int(folds[i]),[]).append((g["_w"],x,y))
    def summary(sample):
        weights,x,y=zip(*sample)
        return {"n":len(x),"base_weighted_ap5":float(np.average(x,weights=weights)),
                "untrained_rrf_weighted_ap5":float(np.average(y,weights=weights))}
    return {"all":summary([item for rows in byfold.values() for item in rows]),"folds":{k:summary(v) for k,v in byfold.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("test","train","groups","corpus","model"):
        p.add_argument("--"+arg,type=Path)
    parent=base.ROOT/"output_2b_final"
    p.add_argument("--lex-dir",type=Path,default=parent/"cpu_rank")
    p.add_argument("--semantic-dir",type=Path,default=parent/"cpu_verified_rank")
    p.add_argument("--weights-dir",type=Path,default=base.ROOT/"weights_2b")
    p.add_argument("--large-weights",type=Path,default=base.ROOT/"weights_2b/cpu_story_ce")
    p.add_argument("--output",type=Path,default=parent/"cpu_large_ce")
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--fixed-answers",type=Path,default=parent/"cpu_verified_cal/oof.csv")
    p.add_argument("--pilot-per-fold",type=int)
    p.add_argument("--depth",type=int,default=32)
    p.add_argument("--facets",action="store_true",
                   help="Also rerank clauses, option statements and original user wording")
    p.add_argument("--nested-reader-contexts",type=Path,
                   help="Compute five sets of nested training rankings for the answer models")
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    if args.depth < 1:
        p.error("--depth must be positive")
    args.corpus=base.find_file(base.ROOT,"corpus.parquet",args.corpus)
    corpus=pd.read_parquet(args.corpus,columns=["post_id","date","channel","text"])
    dates={p:date.fromisoformat(str(d)[:10]) for p,d in zip(corpus.post_id,corpus.date)}
    param=SimpleNamespace(output=args.semantic_dir,rank_dir=args.lex_dir,cache_dir=args.cache_dir,with_e5=True,
                          dense_weights=args.weights_dir/"cpu_dense",e5_weights=args.weights_dir/"cpu_e5")
    engine=semantic.CitationFeatures(param,corpus,args.corpus)
    if not args.model:
        trainfile=base.find_file(base.ROOT,"train.csv",args.train)
        train=base.read_questions(trainfile)
        groups_path=base.find_file(base.ROOT,"train_groups.csv",args.groups)
        folds,splits=base.group_folds(train,groups_path)
        scorer=base.module_from_file(base.find_file(base.ROOT,"score.py"))
        gold={g["id"]:g for g in scorer.load_gold([str(trainfile)])}
        base_ranks=semantic.read_ranks(args.semantic_dir/"train.jsonl")
        if args.pilot_per_fold:
            which=np.concatenate([sorted(va,key=lambda i:hashlib.sha256(train.iloc[i].qid.encode()).digest())[:args.pilot_per_fold] for _,va in splits])
            score=logits("train",train,corpus,args,which)
            report=pilot_scores(train,which,folds,base_ranks,score,gold,scorer)
            (args.output/"pilot.json").write_text(json.dumps(report,indent=2),encoding="utf8");print(report,flush=True);return
        all_i=np.arange(len(train));score=logits("train",train,corpus,args,all_i)
        ids,x=augmented("train",train,engine,dates,param,all_i,score);out=[None]*len(train)
        if args.nested_reader_contexts:
            nested_reader_contexts(train,splits,groups_path,trainfile,ids,x,gold,args)
            return
        for tr,va in splits:
            model=lexical.ranker(x,ids,train,gold,tr,args.threads)
            for i,record in zip(va,lexical.apply(model,train,ids,x,va,args.threads)):out[i]=record
        lexical.dump_jsonl(args.output/"train.jsonl",out)
        report={"base":story.evaluate(scorer,gold,list(base_ranks.values()),args.fixed_answers),
                "large":story.evaluate(scorer,gold,out,args.fixed_answers)}
        (args.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8");print(report,flush=True)
        model=lexical.ranker(x,ids,train,gold,all_i,args.threads);joblib.dump(model,args.output/"model.joblib")
    else:model=joblib.load(args.model)
    test=base.read_questions(base.find_file(base.ROOT,"test.csv",args.test));all_i=np.arange(len(test))
    scores=logits("test",test,corpus,args,all_i);ids,x=augmented("test",test,engine,dates,param,all_i,scores)
    lexical.dump_jsonl(args.output/"test.jsonl",lexical.apply(model,test,ids,x,all_i,args.threads))


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        main()
