"""Nested event reranker and cross-channel evidence expansion on frozen first-stage rankings."""

import argparse
from collections import Counter
from datetime import date
from functools import lru_cache
import json
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import cpu_numeric_2b as numeric
import cpu_rank_2b as lex
import cpu_semantic_2b as sem
import solution_2b_v2 as base


def summaries(sim, near, age, weights):
    """Agreement with distinct corroborating reports across time scales."""
    weighted = np.maximum(sim,0) * weights
    close = weighted * near
    peak = np.argmax(weighted,axis=1)
    row = np.arange(len(weighted))
    return [sim[:,0], weighted[:,:3].max(axis=1), weighted.max(axis=1), weighted.sum(axis=1),
            close.max(axis=1), close.sum(axis=1), (sim*near).max(axis=1),
            (weighted/(1+age/30.)).max(axis=1), np.log1p(age[row,peak]),
            np.minimum(8,(sim > .6).sum(axis=1)), np.minimum(8,((sim > .6)*near).sum(axis=1))]


class Stories:
    def __init__(self, corpus, corpus_path, params):
        docs = corpus.sort_values(["date","post_id"])
        self.ids = docs.post_id.to_numpy(dtype=np.int64)
        self.pos = {int(p):i for i,p in enumerate(self.ids)}
        self.days = docs.date.to_numpy("datetime64[D]").astype(np.int32)
        self.text = docs.text.fillna("").to_numpy()
        self.channels = pd.factorize(docs.channel)[0]
        self.vectors = [np.load(sem.embedding_path(params,corpus_path,path),mmap_mode="r")
                        for path in [params.dense_weights,params.e5_weights]]

    @lru_cache(maxsize=120000)
    def terms(self, doc):
        return frozenset(numeric._words(str(self.text[doc]).lower().replace("ё","е")))

    def expand(self, anchors, hi):
        parts=[]
        for anchor in anchors[:8]:
            d=self.days[anchor]
            lo=self.days[:hi].searchsorted(d-2)
            end=self.days[:hi].searchsorted(d+2,side="right")
            window=np.arange(lo,end,dtype=np.int32)
            window=window[self.channels[window]!=self.channels[anchor]]
            if not len(window):
                continue
            score = sum(mat[window].astype(np.float32) @ mat[anchor].astype(np.float32) for mat in self.vectors)
            score /= 2
            take=lex.best(score,5)
            parts.extend(window[take].tolist())
        return parts

    def features(self,row,ids,scores):
        cutoff = np.datetime64(row["asked_at"],"D").astype(np.int32)
        hi = self.days.searchsorted(cutoff,side="right")
        order=np.lexsort((ids,-scores))
        original=np.asarray([self.pos[int(p)] for p in ids[order]],np.int32)
        if np.any(original>=hi):
            raise ValueError("Future first-stage anchor")
        if not len(original):
            return np.array([],np.int64),None,np.array([],np.float32)
        raw = scores[order].astype(np.float32)
        anchors = original[:20]
        docs=np.array(list(dict.fromkeys([*original.tolist(),*self.expand(anchors,hi)])),np.int32)
        published=self.days[docs]
        if np.any(published>cutoff):
            raise ValueError("Future event candidate")
        n=len(docs)
        prior=np.full(n, float(raw[0])-4.0,np.float32);prior[:len(raw)]=raw
        ranks=np.zeros(n);ranks[:len(raw)]=1/(1+np.arange(len(raw)))
        # The base margin is only comparable within a question. All weights use that question's anchors.
        weight=np.exp(np.minimum(0,.65*(raw[:len(anchors)]-raw[0])))
        diff=np.abs(published[:,None]-self.days[anchors][None,:])
        not_self=docs[:,None]!=anchors[None,:]
        near=(diff<=2) & (self.channels[docs,None]!=self.channels[anchors][None,:])
        features=[prior,prior-raw[0],ranks,np.arange(n)>=len(raw),np.log1p(cutoff-published),
                  np.full(n,raw[0]),np.full(n,np.median(raw[:min(5,len(raw))]))]
        for vectors in self.vectors:
            sim=(vectors[docs].astype(np.float32) @ vectors[anchors].astype(np.float32).T)*not_self
            features.extend(summaries(sim,near,diff,weight))
        word_sets=[self.terms(int(d)) for d in docs]
        counts=Counter(w for terms in word_sets for w in terms)
        # Local IDF uses only documents in the visible candidate set.
        qwords=set(numeric._words(row["question"].lower().replace("ё","е")))
        lookup={w:(j,1+np.log((n+1)/(count+1))) for j,(w,count) in enumerate(sorted(counts.items()))}
        iv=[]; vals=[]; offsets=[0]; signature=[]; qcoverage=[]; numcover=[]
        qnums={w for w in qwords if w.isdigit()}
        total=sum(lookup[w][1]**2 for w in qwords if w in lookup) or 1
        for d,terms in zip(docs,word_sets):
            shared=qwords & terms
            qcoverage.append(sum(lookup[w][1]**2 for w in shared)/total)
            numcover.append(len(terms & qnums)/max(1,len(qnums)))
            key=[lookup[w] for w in terms]
            norm=(sum(v*v for _,v in key)**.5) or 1.
            for j,v in key:
                iv.append(j);vals.append(v/norm)
            offsets.append(len(iv))
            signature.append(len(terms))
        matrix=csr_matrix((np.array(vals,np.float32),iv,offsets),shape=(n,len(lookup)))
        sim=(matrix @ matrix[:len(anchors)].T).toarray()*not_self
        features.extend(summaries(sim,near,diff,weight))
        occurrence=(matrix.sign() @ matrix[:len(anchors)].sign().T).toarray()*not_self
        features.extend([occurrence.max(axis=1), (occurrence*near).max(axis=1),
                         (occurrence*weight).max(axis=1),qcoverage,numcover,np.log1p(signature),
                         (near*weight).sum(axis=1), (not_self/(1+diff/21.)*weight).sum(axis=1)])
        targets=lex.targets(row["question"],row["asked_at"])
        distance=np.min(np.abs(published[:,None]-targets[None,:]),axis=1) if len(targets) else np.full(n,2000)
        features.extend([np.log1p(distance),np.full(n,len(targets)),np.full(n,len(qwords))])
        # Evidence types and temporal questions need different treatment of old events.
        features.extend(np.full(n,row["answer_format"]==fmt) for fmt in lex.FORMATS)
        features.append(np.full(n, numeric._clauses(row["question"])[0]!=numeric._clauses(row["question"])[1]))
        return self.ids[docs],np.column_stack(features).astype(np.float32),prior


def learn(features, gold, frame, train, threads):
    from lightgbm import LGBMRanker
    xx=[]; yy=[]; ww=[]; prior=[]; sizes=[]
    for i in train:
        ids,x,p=features[i]
        if not len(ids):
            continue
        g=gold[frame.iloc[i].qid]
        y=np.isin(ids,g["_G"]).astype(np.int8)
        if not y.any():
            continue
        xx.append(x);yy.append(y);sizes.append(len(y));prior.extend(p)
        ww.extend([g["_w"]]*len(y))
    return LGBMRanker(n_estimators=100,num_leaves=11,learning_rate=.025,min_child_samples=65,
                      reg_lambda=12,lambdarank_truncation_level=8,random_state=123,n_jobs=threads,
                      deterministic=True,force_col_wise=True,verbosity=-1).fit(
                          np.vstack(xx),np.concatenate(yy),group=sizes,init_score=np.asarray(prior),sample_weight=ww)


def apply(model, features, frame, selected, threads):
    records=[]
    for i in selected:
        ids,x,p=features[i]
        score=(p+model.booster_.predict(x,num_threads=threads)) if len(ids) else p
        records.append(lex.prediction(frame.iloc[i].qid,ids,score))
    return records


def fixed_features(engine, frame, ranks):
    features=[]
    for row in frame.to_dict("records"):
        item=ranks[row["qid"]]
        features.append(engine.features(row,np.array(item["ids"],np.int64),np.array(item["scores"],np.float32)))
    return features


def evaluate(scorer, gold, rankings, answer_path):
    from csv import DictReader
    with answer_path.open(encoding="utf8",newline="") as f:
        answers={r["qid"]:r["answer"] for r in DictReader(f)}
    if set(answers)!=set(gold):
        raise ValueError("Fixed answer OOF qids differ from training data")
    metrics={}
    for level in ("all","light","medium","heavy"):
        cases=[(gold[r["qid"]],r["ids"]) for r in rankings if level=="all" or gold[r["qid"]]["level"]==level]
        preds=[scorer.score_item(g,{"answer":answers[g["id"]],"evidence":ids[:5] if answers[g["id"]]!="no_info" else []}) for g,ids in cases]
        oracle=[scorer.score_item(g,{"answer":g["answer"],"evidence":ids[:5] if g["answer"]!="no_info" else []}) for g,ids in cases]
        nz=[(g,ids) for g,ids in cases if g["_G"]];weights=[g["_w"] for g,_ in nz]
        metrics[level]={"S_fixed_answers":scorer.total(preds),"S_golden_answers":scorer.total(oracle),
                        "weighted_ap5":float(np.average([scorer.ap5(ids[:5],g["_G"]) for g,ids in nz],weights=weights)),
                        "hit5":float(np.mean([bool(set(g["_G"])&set(ids[:5])) for g,ids in nz]))}
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("train","test","corpus","groups","model"):
        p.add_argument("--"+name,type=Path)
    base_dir=base.ROOT/"output_2b_final"
    p.add_argument("--lex-dir",type=Path,default=base_dir/"cpu_rank")
    p.add_argument("--semantic-dir",type=Path,default=base_dir/"cpu_verified_rank")
    p.add_argument("--weights-dir",type=Path,default=base.ROOT/"weights_2b")
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--fixed-answers",type=Path,default=base_dir/"cpu_verified_cal/oof.csv")
    p.add_argument("--output",type=Path,default=base_dir/"cpu_story_rank")
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    corpus_path=base.find_file(base.ROOT,"corpus.parquet",a.corpus)
    corpus=pd.read_parquet(corpus_path,columns=["post_id","date","channel","text"])
    params=SimpleNamespace(output=a.semantic_dir,rank_dir=a.lex_dir,with_e5=True,cache_dir=a.cache_dir,
                           dense_weights=a.weights_dir/"cpu_dense",e5_weights=a.weights_dir/"cpu_e5")
    engine=Stories(corpus,corpus_path,params)
    if a.model:
        model=joblib.load(a.model)
    else:
        train_path=base.find_file(base.ROOT,"train.csv",a.train)
        frame=base.read_questions(train_path);groups_path=base.find_file(base.ROOT,"train_groups.csv",a.groups)
        folds,splits=base.group_folds(frame,groups_path)
        groups=frame[["qid"]].merge(pd.read_csv(groups_path,dtype=str),on="qid",validate="one_to_one")["group"].to_numpy()
        scorer=base.module_from_file(base.find_file(base.ROOT,"score.py"))
        gold={r["id"]:r for r in scorer.load_gold([str(train_path)])}
        saved=sem.read_ranks(a.semantic_dir/"train.jsonl")
        dates={p:date.fromisoformat(str(d)[:10]) for p,d in zip(corpus.post_id,corpus.date)}
        first=joblib.load(a.semantic_dir/"model.joblib")
        cite=sem.CitationFeatures(params,corpus,corpus_path) if first.n_features_in_>lex.FEATURE_COUNT+22 else None
        raw_ids,raw_x=sem.inputs("train",frame.assign(_day=frame.asked_at.apply(date.fromisoformat)),params,dates,cite)
        if raw_x[0].shape[1]!=first.n_features_in_:
            raise ValueError("First-stage feature schema differs from saved ranker")
        real=fixed_features(engine,frame,saved)
        output=[None]*len(frame); inner=GroupKFold(5)
        for fold,(tr,va) in enumerate(splits):
            since=time.monotonic(); features={}
            for aidx,bidx in inner.split(tr,groups=groups[tr]):
                train_i,held_i=tr[aidx],tr[bidx]
                learned=lex.ranker(raw_x,raw_ids,frame,gold,train_i,a.threads)
                for i in held_i:
                    score=learned.booster_.predict(raw_x[i],num_threads=a.threads)
                    features[i]=engine.features(frame.iloc[i].to_dict(),raw_ids[i],score)
            model=learn(features,gold,frame,tr,a.threads)
            for i,r in zip(va,apply(model,real,frame,va,a.threads)):
                output[i]=r
            print(f"Nested fold {fold+1}/5 complete in {time.monotonic()-since:.1f}s",flush=True)
        lex.dump_jsonl(a.output/"train.jsonl",output)
        report={"base":evaluate(scorer,gold,list(saved.values()),a.fixed_answers),
                "story":evaluate(scorer,gold,output,a.fixed_answers),"folds":{}}
        for idx in range(5):
            gs={q:g for i,(q,g) in enumerate(gold.items()) if folds[i]==idx}
            report["folds"][idx]=[(r["qid"],r["ids"][:5]) for i,r in enumerate(output) if folds[i]==idx]
            report["folds"][idx]={"n":len(gs),
                "weighted_ap5":float(np.average([scorer.ap5(ids,gs[q]["_G"]) for q,ids in report["folds"][idx] if gs[q]["_G"]],
                                               weights=[gs[q]["_w"] for q,_ in report["folds"][idx] if gs[q]["_G"]]))}
        (a.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
        print(report["story"]["all"],flush=True)
        model=learn(real,gold,frame,range(len(frame)),a.threads)
        joblib.dump(model,a.output/"model.joblib")
    test=base.read_questions(base.find_file(base.ROOT,"test.csv",a.test))
    feat=fixed_features(engine,test,sem.read_ranks(a.semantic_dir/"test.jsonl"))
    lex.dump_jsonl(a.output/"test.jsonl",apply(model,feat,test,range(len(test)),a.threads))


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        main()
