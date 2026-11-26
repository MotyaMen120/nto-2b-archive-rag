"""Fit a grouped domain-specific evidence scorer on frozen cross-encoder representations."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import onnx
from onnx import TensorProto, helper
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import cpu_rank_2b as rank
import cpu_semantic_2b as semantic
import solution_2b_v2 as base


def graph(weights, working):
    name="probe_" + semantic.digest(weights/"model.onnx")[:14]
    dest=working/name
    dest.mkdir(parents=True,exist_ok=True)
    model=dest/"model.onnx"
    if not model.is_file():
        network=onnx.load(weights/"model.onnx")
        indices="probe_zero"
        network.graph.initializer.append(helper.make_tensor(indices,TensorProto.INT64,[],[0]))
        outputs=["/classifier/Gather_output_0","/classifier/Tanh_output_0"]
        for layer in (5,9):
            hidden=f"/roberta/encoder/layer.{layer}/output/LayerNorm/Add_1_output_0"
            output=f"probe_cls_{layer}"
            network.graph.node.append(helper.make_node("Gather",[hidden,indices],[output],axis=1))
            outputs.append(output)
        network.graph.node.append(helper.make_node("Concat",outputs,["probe_vector"],axis=1))
        network.graph.output.append(helper.make_tensor_value_info("probe_vector",TensorProto.FLOAT,[None,1536]))
        temporary=dest/"model.tmp"
        onnx.save(network,temporary)
        temporary.replace(model)
        (dest/"tokenizer.json").write_bytes((weights/"tokenizer.json").read_bytes())
    return dest


def extract(args,frame,split,corpus,corpus_path):
    import hashlib
    ce_path=args.rank_dir/f"ce_{split}.jsonl"
    selections=semantic.read_ranks(ce_path)
    weights=graph(args.weights,args.output/"cache")
    engine=semantic.Engine(weights,args.threads,192)
    engine.output="probe_vector"
    cache_key=semantic.digest(ce_path)+semantic.digest(corpus_path)+semantic.digest(weights/"model.onnx")+semantic.digest(weights/"tokenizer.json")
    dates=dict(zip(corpus.post_id,corpus.date.astype(str)))
    texts=corpus.set_index("post_id").text.to_dict()
    reps=[]
    start=time.perf_counter()
    for i,row in enumerate(frame.to_dict("records"),1):
        request=base.rewrite(row)["canonical_query"]
        ids=selections[row["qid"]]["ids"]
        if any(dates[p][:10]>row["asked_at"] for p in ids):
            raise ValueError("Future pair before representation extraction")
        key=hashlib.sha256(json.dumps([cache_key,row["qid"],row["asked_at"],request,ids]).encode()).hexdigest()
        path=args.output/"cache"/f"{key}.npy"
        if path.is_file():
            x=np.load(path,allow_pickle=False)
        elif ids:
            x=engine.encode([(request,texts[p]) for p in ids],8,"probe").astype(np.float16)
            tmp=path.with_suffix(".part.npy")
            np.save(tmp,x,allow_pickle=False)
            tmp.replace(path)
        else:
            x=np.empty((0,1536),np.float16)
        reps.append(x)
        if i%80==0:
            print(f"{split} probe {i}/{len(frame)} in {time.perf_counter()-start:.0f}s",flush=True)
    return reps,[np.array(selections[q]["ids"],dtype=np.int64) for q in frame.qid]


def fit_probe(ids,vectors,train,gold,indices):
    x=[]
    y=[]
    weights=[]
    rng=np.random.default_rng(192)
    for i in indices:
        row=gold[train.iloc[i].qid]
        positives=np.isin(ids[i],row["_G"])
        pos=np.flatnonzero(positives)
        neg=np.flatnonzero(~positives)
        if not len(pos) or not len(neg):
            continue
        matrix=vectors[i].astype(np.float32)
        for a in pos:
            hard=neg[:10]
            random=rng.choice(neg,min(12,len(neg)),replace=False)
            rivals=np.unique(np.r_[hard,random])
            pairs=(matrix[a]-matrix[rivals])*25
            x.extend([pairs,-pairs])
            y.extend([np.ones(len(pairs),np.int8),np.zeros(len(pairs),np.int8)])
            weights.extend([row["_w"]/(len(pos)*len(rivals))]*2*len(rivals))
    classifier=LogisticRegression(C=.09,fit_intercept=False,max_iter=1000,solver="liblinear",
                                  random_state=192)
    classifier.fit(np.vstack(x),np.concatenate(y),sample_weight=weights)
    return classifier


def rerank(classifier, qid, original, ids, vectors):
    order=original["ids"]
    scores=np.asarray(original["scores"],np.float64)
    evidence=list(ids)
    if len(evidence):
        probe=classifier.decision_function(vectors.astype(np.float32)*25)
        # Keep the base score for candidates outside the frozen text shortlist.
        median=float(np.median(probe))
        deltas={int(p):float(v)-median for p,v in zip(ids,probe)}
        scores=scores+np.array([deltas.get(p,0.) for p in order])*.55
    return rank.prediction(qid,np.asarray(order,dtype=np.int64),scores)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("corpus","train","test","groups","model"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--rank-dir",type=Path,default=base.ROOT/"output_2b_final/cpu_verified_rank")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--weights",type=Path,default=base.ROOT/"weights_2b/cpu_ce")
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cp=base.find_file(base.ROOT,"corpus.parquet",args.corpus)
    corpus=pd.read_parquet(cp,columns=["text","date","post_id"])
    with threadpool_limits(limits=1,user_api="blas"):
        if args.model:
            classifier=joblib.load(args.model)
        else:
            tp=base.find_file(base.ROOT,"train.csv",args.train)
            train=base.read_questions(tp)
            _,splits=base.group_folds(train,base.find_file(base.ROOT,"train_groups.csv",args.groups))
            scorer=base.module_from_file(base.find_file(base.ROOT,"score.py"))
            gold={r["id"]:r for r in scorer.load_gold([str(tp)])}
            vectors,ids=extract(args,train,"train",corpus,cp)
            original=semantic.read_ranks(args.rank_dir/"train.jsonl")
            preds=[None]*len(train)
            for tr,va in splits:
                classifier=fit_probe(ids,vectors,train,gold,tr)
                for i in va:
                    q=train.iloc[i].qid
                    preds[i]=rerank(classifier,q,original[q],ids[i],vectors[i])
            rank.dump_jsonl(args.output/"train.jsonl",preds)
            def measure(rows):
                rows=[r for r in rows if gold[r["qid"]]["_G"]]
                w=[gold[r["qid"]]["_w"] for r in rows]
                return float(np.average([scorer.ap5(r["ids"][:5],gold[r["qid"]]["_G"]) for r in rows],weights=w))
            report={"base_weighted_ap5":measure(list(original.values())),"probe_weighted_ap5":measure(preds)}
            print(report,flush=True)
            (args.output/"evidence.report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
            classifier=fit_probe(ids,vectors,train,gold,range(len(train)))
            joblib.dump(classifier,args.output/"model.joblib")
            del vectors
        test=base.read_questions(base.find_file(base.ROOT,"test.csv",args.test))
        vectors,ids=extract(args,test,"test",corpus,cp)
        original=semantic.read_ranks(args.rank_dir/"test.jsonl")
        records=[rerank(classifier,q,original[q],ids[i],vectors[i]) for i,q in enumerate(test.qid)]
        rank.dump_jsonl(args.output/"test.jsonl",records)


if __name__=="__main__":
    main()
