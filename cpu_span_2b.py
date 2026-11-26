"""Batched, offline SberQuAD answer-span features and fold-isolated answer selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
from threadpoolctl import threadpool_limits

import cpu_numeric_2b as numeric
import solution_2b_v2 as base


REVISION = "5d5d39ea0f22892203dc47c4120f5e1a55252677"
MODEL_SHA = "8daa21693ac0ac7a78dd49b92d1128d0fbba1c0f4e50164723ebc6600e6d3986"
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()


def _rubric(q):
    unit = numeric._norm(q.get("unit") or "")
    if q["answer_format"]=="date":
        return "first" if numeric.FIRST_NEWS.search(numeric._norm(q["question"])) else "date"
    if numeric._is_timeline(numeric._norm(q["question"]),unit):
        return "time"
    return "quantity"


def _specs(row,posts,k):
    """Propose grounded spans. Offsets refer to the exact same passage the model reads."""
    kind = _rubric(row)
    docs = numeric._documents(row,posts,limit=60 if kind=="time" else k)
    orig = numeric._norm(row["question"])
    if kind=="time":
        clauses = numeric._clauses(orig)
        tasks = []
        for role,clause in enumerate(clauses):
            anchors = numeric._event_dates(clause,clauses[1-role],docs)
            indices = list(dict.fromkeys(anchor[2] for anchor in anchors))[:k]
            news = bool(numeric.NEWS.search(clause))
            question = ("Когда опубликовали новость" if news else "Когда произошло событие") + " «" + clause.strip(" ,.?") + "»?"
            tasks.extend((role,rank,question) for rank in indices)
    else:
        tasks = [(0,rank,row["question"]) for rank in range(len(docs))]
    for role,rank,question in tasks:
        doc=docs[rank]
        pub=doc.pub
        prefix=f"Дата публикации: {pub.day} {MONTHS[pub.month-1]} {pub.year} года. "
        entries=[]
        if kind=="quantity":
            focus=numeric._words(orig)
            for m in numeric._mentions(doc,focus,focus,numeric._norm(row.get("unit") or "")):
                entries.append((numeric._number(m.value),len(prefix)+m.start,len(prefix)+m.end,m.hint))
        else:
            entries.append((pub.isoformat(),prefix.index(":")+2,prefix.index(" года"),1.0))
            if kind!="first":
                for m in doc.moments:
                    entries.append((m.value.isoformat(),len(prefix)+m.start,len(prefix)+m.end,0.0))
        if entries:
            yield role,rank,question,prefix+doc.text,entries


class SpanEngine:
    def __init__(self,directory,cache,threads=2,length=256,batch_size=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        model=directory/"model.onnx"
        with model.open("rb") as f:
            digest=hashlib.file_digest(f,"sha256").hexdigest()
        if digest!=MODEL_SHA:
            raise ValueError("SberQuAD ONNX checksum differs from the pinned model")
        options=ort.SessionOptions()
        options.intra_op_num_threads=threads
        options.inter_op_num_threads=1
        options.add_session_config_entry("session.intra_op.allow_spinning","0")
        self.net=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
        self.outputs=[a.name for a in self.net.get_outputs()]
        self.inputs={a.name for a in self.net.get_inputs()}
        self.tokenizer=Tokenizer.from_file(str(directory/"tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=length)
        self.tokenizer.no_padding()
        self.pad=self.tokenizer.token_to_id("<pad>")
        self.batch_size=batch_size
        self.key=f"{digest}|{length}|offset-v2|".encode()
        self.cache=sqlite3.connect(cache)
        self.cache.execute("CREATE TABLE IF NOT EXISTS qa(key TEXT PRIMARY KEY, result TEXT NOT NULL)")

    def _run_batch(self, texts):
        tokenized=self.tokenizer.encode_batch([(q,p) for q,p in texts])
        width=max(len(t.ids) for t in tokenized)
        ids=np.full((len(texts),width),self.pad,np.int64)
        mask=np.zeros_like(ids)
        types=np.zeros_like(ids)
        for i,tok in enumerate(tokenized):
            n=len(tok.ids)
            ids[i,:n]=tok.ids
            mask[i,:n]=1
            types[i,:n]=tok.type_ids
        data={"input_ids":ids,"attention_mask":mask,"token_type_ids":types}
        outs=self.net.run(self.outputs,{key:data[key] for key in self.inputs})
        start,end=(np.asarray(o) for o in outs)
        results=[]
        for i,token in enumerate(tokenized):
            indices=[j for j,s in enumerate(token.sequence_ids) if s==1 and token.offsets[j][1]>token.offsets[j][0]]
            if not indices:
                results.append([])
                continue
            starts=start[i,indices]
            ends=end[i,indices]
            start_logp=starts-np.logaddexp.reduce(starts)
            end_logp=ends-np.logaddexp.reduce(ends)
            results.append([[*token.offsets[j],round(float(a),3),round(float(b),3)]
                            for j,a,b in zip(indices,start_logp,end_logp)])
        return results

    def fetch(self,texts,qid,cutoff):
        # Quantized kernels may choose activation scales across a batch. Each batch
        # contains only the eligible posts of one question and has a stable order.
        keyed={text:hashlib.sha256(self.key+(text[0]+"\0"+text[1]).encode()).hexdigest() for text in texts}
        found={}
        ordered=sorted(keyed,key=lambda t:(len(t[0])+len(t[1]),t))
        for off in range(0,len(ordered),self.batch_size):
            group=ordered[off:off+self.batch_size]
            signature=hashlib.sha256((qid+"|"+cutoff+"|"+"|".join(keyed[t] for t in group)).encode()).hexdigest()
            cache_keys={t:signature+"|"+keyed[t] for t in group}
            existing=[self.cache.execute("SELECT result FROM qa WHERE key=?",(cache_keys[t],)).fetchone() for t in group]
            if all(existing):
                found.update((t,json.loads(row[0])) for t,row in zip(group,existing))
                continue
            for text,answer in zip(group,self._run_batch(group)):
                found[text]=answer
                self.cache.execute("INSERT OR REPLACE INTO qa VALUES(?,?)",(cache_keys[text],json.dumps(answer,separators=(",",":"))))
            self.cache.commit()
        return found


def _span_support(tokens,start,end):
    indices=[j for j,(a,b,_,_) in enumerate(tokens) if a < end and b > start]
    if not indices:
        return None
    i,j=indices[0],indices[-1]
    starters=range(max(0,i-2),min(j+1,i+3))
    best=-80.0
    for head in starters:
        for tail in range(max(head,j-2),min(len(tokens),j+4)):
            # Span heads may include "около" and tails can include the noun for a measure.
            best=max(best,tokens[head][2]+tokens[tail][3] - .22*abs(i-head)-.12*abs(j-tail))
    return best


def _summarize(raw):
    if not raw:
        return {}
    grouped=defaultdict(dict)
    for value,score,rank,hint in raw:
        prev=grouped[value].get(rank)
        if prev is None or score>prev[0]:
            grouped[value][rank]=(score,hint)
    features={}
    for value,ranks in grouped.items():
        scored=[(s-.60*math.log1p(rank),s,rank,hint) for rank,(s,hint) in ranks.items()]
        adjusted,raw,rank,hint=max(scored)
        votes=sum(math.exp(max(-20,s-adjusted)) for s,_,_,_ in scored)
        features[value]=[adjusted+.65*math.log(votes),raw,1/(1+rank),hint,
                         math.log1p(len(scored)),math.log(votes)]
    ordered=sorted(features,key=lambda value:-features[value][0])
    for pos,v in enumerate(ordered):
        features[v].extend([1/(1+pos),features[v][0]-features[ordered[0]][0]])
    return features


def _span_features(row,specs,computed):
    by_role=defaultdict(list)
    for role,rank,question,passage,entries in specs:
        tokens=computed[(question,passage)]
        for value,start,end,hint in entries:
            score=_span_support(tokens,start,end)
            if score is not None:
                by_role[role].append((value,score,rank,hint))
    if _rubric(row)!="time":
        return _summarize(by_role[0])
    ends=[sorted(_summarize(by_role[i]).items(),key=lambda kv:-kv[1][0])[:10] for i in (0,1)]
    hypotheses=[]
    for a,sa in ends[0]:
        for b,sb in ends[1]:
            if b<=a:
                continue
            da,db=map(numeric.date.fromisoformat,(a,b))
            if "месяц" in numeric._norm(row["unit"]):
                n=(db.year-da.year)*12+db.month-da.month-int("полн" in row["question"] and db.day<da.day)
            else:
                n=(db-da).days
            if n>0:
                score=(sa[0]+sb[0])/2
                hypotheses.append((str(n),score,int(1/min(sa[2],sb[2]))-1,min(sa[3],sb[3])))
    return _summarize(hypotheses)


def build(rows,rankings,corpus,engine,k):
    out={}
    for off in range(0,len(rows),16):
        chunk=[]
        for row in rows[off:off+16]:
            if row["answer_format"] not in {"number","date"}:
                continue
            rank=rankings[row["qid"]]
            posts=[corpus[int(k)] for k in rank["ids"][:60]]
            if any(str(p["date"])[:10]>str(row["asked_at"])[:10] for p in posts):
                raise ValueError(f"Future candidate for {row['qid']}")
            specs=list(_specs(row,posts,k))
            chunk.append((row,posts,specs))
        for row,posts,specs in chunk:
            computed=engine.fetch([(q,p) for _,_,q,p,_ in specs],row["qid"],str(row["asked_at"])[:10])
            spans=_span_features(row,specs,computed)
            base_cand=numeric._candidates(row,posts)
            combined={}
            kind=_rubric(row)
            universe=set(spans)|set(base_cand)
            if kind=="quantity":
                universe.add("no_info")
            elif kind=="date":
                universe.add("promised")
            top=sorted(spans.values(),key=lambda f:-f[0])
            reliability=[top[0][1],top[0][0]-top[1][0] if len(top)>1 else 1.0,top[0][4]] if top else [-40,0,0]
            for v in universe:
                src=base_cand.get(v)
                prior=[src[0],src[1],src[2],src[4],src[6],src[7],src[-2],src[-1]] if src else [-15,0,0,0,0,0,-15,0]
                neural=spans.get(v,[-40,-40,0,0,0,0,0,-40])
                combined[v]=[*neural,*prior,*[float(kind==a) for a in ("quantity","date","first","time")],
                             float(v in ("promised","no_info")),float("руб" in str(row["unit"])),*reliability]
            out[row["qid"]]=combined
        if off%160==0:
            print(f"Span contexts {off}/{len(rows)}",flush=True)
    return out


def _label(value,answer,fmt):
    return int(numeric._match(value,str(answer),fmt))


def train_selector(rows,features):
    from sklearn.ensemble import HistGradientBoostingClassifier
    x,y,weights=[],[],[]
    for row in rows:
        cand=features.get(row["qid"],{})
        yes=[v for v in cand if _label(v,row["answer"],row["answer_format"])]
        if not yes:
            continue
        # Include the highest-scoring distractors from each of the two readers.
        no=[v for v in cand if v not in yes]
        no=set(sorted(no,key=lambda v:(-cand[v][0],v))[:14]+sorted(no,key=lambda v:(-cand[v][8],v))[:14])
        for group,label,weight in ((yes,1,1.0),(sorted(no),0,2.0)):
            for value in group:
                x.append(cand[value])
                y.append(label)
                weights.append(weight/len(group))
    model=HistGradientBoostingClassifier(max_iter=80,learning_rate=.07,max_leaf_nodes=8,
                                         min_samples_leaf=14,l2_regularization=2,
                                         random_state=53,early_stopping=False)
    with threadpool_limits(limits=1):
        model.fit(x,y,sample_weight=weights)
    return model


def select(model,row,features,incumbent):
    if isinstance(model,dict):
        candidate=select(model["selector"],row,features,incumbent)
        if candidate==incumbent or model["gate"] is None:
            return incumbent
        return candidate if model["gate"].predict([_gate_features(row,features,incumbent,candidate)])[0]>0 else incumbent
    if incumbent in {"no_info","promised"} or not features:
        return incumbent
    vals=sorted(features)
    with threadpool_limits(limits=1):
        logits=model.decision_function([features[v] for v in vals])
    # Direct extraction supplies an independent signal only for an answer grounded
    # in one of the candidate spans. Keep the incumbent for pure arithmetic guesses.
    scores=[]
    for v,logit in zip(vals,logits):
        f=features[v]
        special=v in {"no_info","promised"}
        prior=0 if special else .20*max(-5,f[14]) if f[14]>=-10 else -.8
        span=0 if special else .15*max(-6,f[7]) if f[7]>=-10 else -.65
        scores.append(float(logit)+prior+span+.55*(v==incumbent))
    best=vals[int(np.argmax(scores))]
    if features[best][6]<=0 and best not in {"no_info","promised"}:
        return incumbent
    return best


def _gate_features(row,features,incumbent,alternative):
    v=features[alternative]
    old=features.get(incumbent)
    return [float(_rubric(row)==kind) for kind in ("quantity","date","first","time")]+[
        float(incumbent=="skip"),float(alternative in ("promised","no_info")),
        float(bool(numeric.re.search(r"разниц|больш|меньш|дешев|дороже|вместе|процент",numeric._norm(row["question"])))),
        np.clip(v[0],-10,0)/5,np.clip(v[4],0,2),v[6],np.clip(v[7],-6,0)/3,
        np.clip(v[14],-8,0)/4,
        np.clip(v[0]-(old[0] if old else -40),-8,8)/4,
        np.clip(v[8]-(old[8] if old else -15),-6,6)/3]


def train_gated(rows,features,corpus,seed,seed_posts,scorer,gold,groups,prepared):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    import cpu_abstention_2b as abstain
    subset=[r for r in rows if r["answer_format"] in {"date","number"}]
    x,y,weights=[],[],[]
    for tr,va in GroupKFold(5).split(subset,groups=[groups[q["qid"]] for q in subset]):
        train=[subset[i] for i in tr]
        span=train_selector(train,features)
        base_reader=numeric.fit_ranker(train,corpus,seed)
        policy=abstain.fit(train,prepared,gold)
        for i in va:
            q=subset[i]
            qid=q["qid"]
            posts=seed_posts[qid]
            incumbent=numeric.predict(q,posts,base_reader)
            if abstain.prefer_skip(policy,q,posts,incumbent):
                incumbent="skip"
            alternative=select(span,q,features[qid],incumbent)
            if alternative==incumbent:
                continue
            def utility(answer):
                ev=[] if answer=="no_info" else [p["post_id"] for p in posts[:5]]
                return scorer.score_item(gold[qid],dict(answer=answer,evidence=ev))["s"]
            x.append(_gate_features(q,features[qid],incumbent,alternative))
            y.append(utility(alternative)-utility(incumbent))
            weights.append(gold[qid]["_w"])
    gate=None
    if len(x)>=15:
        gate=Ridge(alpha=22)
        w=np.asarray(weights)
        gate.fit(x,y,sample_weight=w/w.mean())
    return {"selector":train_selector(rows,features),"gate":gate}


def _records(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def main():
    import joblib
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    from run_cpu_2b import write_checked, evaluate

    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("corpus","train","test","groups"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--rank-dir",type=Path,required=True)
    parser.add_argument("--seed-dir",type=Path,required=True)
    parser.add_argument("--base-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weights",type=Path,default=base.ROOT/"weights_2b/cpu_span")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--top-k",type=int,default=3)
    parser.add_argument("--cache",type=Path,default=base.ROOT/"output_2b_final/cpu_span/cache.sqlite")
    parser.add_argument("--model",type=Path,help="Predict test only with a fitted selector")
    parser.add_argument("--gate",action="store_true",help="Estimate span versus incumbent utility using inner GroupKFold")
    args=parser.parse_args()
    started=time.perf_counter()
    out=args.output.parent
    out.mkdir(parents=True,exist_ok=True)
    args.cache.parent.mkdir(parents=True,exist_ok=True)
    test_file=base.find_file(base.ROOT,"test.csv",args.test)
    corpus_file=base.find_file(base.ROOT,"corpus.parquet",args.corpus)
    test=_records(test_file)
    def ranks(path):
        with path.open(encoding="utf8") as f:
            return {r["qid"]:r for line in f if (r:=json.loads(line))}
    test_rank=ranks(args.rank_dir/"test.jsonl")
    sets=[(test,test_rank)]
    if not args.model:
        train_file=base.find_file(base.ROOT,"train.csv",args.train)
        train=_records(train_file)
        train_rank=ranks(args.rank_dir/"train.jsonl")
        seed=ranks(args.seed_dir/"seed_train.jsonl")
        sets.extend([(train,train_rank),(train,seed)])
    required={int(pid) for qs,rmap in sets for q in qs if q["answer_format"] in {"date","number"}
              for pid in rmap[q["qid"]]["ids"][:60]}
    if not args.model and args.gate:
        required.update(int(pid) for q in train if q["answer_format"] in {"date","number"} for pid in q["evidence"].split())
    wanted=pa.array(list(required))
    corpus={}
    for batch in pq.ParquetFile(corpus_file).iter_batches(batch_size=4096,columns=["post_id","text","date","channel","channel_type"]):
        for post in batch.filter(pc.is_in(batch.column("post_id"),value_set=wanted)).to_pylist():
            post["date"]=str(post["date"])[:10]
            corpus[int(post["post_id"])]=post
    engine=SpanEngine(args.weights,args.cache,args.threads,256,args.batch_size)
    def solve(q,baseline,feats,model):
        answer=select(model,q,feats.get(q["qid"],{}),baseline["answer"])
        return dict(baseline,answer=answer,evidence="" if answer=="no_info" else baseline["evidence"])
    if args.model:
        model=joblib.load(args.model)
    else:
        print("Independent seed features",flush=True)
        seed_features=build(train,seed,corpus,engine,args.top_k)
        print("OOF features",flush=True)
        oof_features=build(train,train_rank,corpus,engine,args.top_k)
        baseline={r["qid"]:r for r in _records(args.base_dir/"oof.csv")}
        train_frame=base.read_questions(train_file)
        groups_file=base.find_file(base.ROOT,"train_groups.csv",args.groups)
        folds,splits=base.group_folds(train_frame,groups_file)
        if args.gate:
            import cpu_abstention_2b as abstain
            groups={r["qid"]:r["group"] for r in _records(groups_file)}
            scorer=base.module_from_file(base.find_file(base.ROOT,"score.py"))
            gold={r["id"]:r for r in scorer.load_gold([str(train_file)])}
            seed_posts={q["qid"]:[corpus[int(k)] for k in seed[q["qid"]]["ids"][:60]]
                        for q in train if q["answer_format"] in ("number","date")}
            prepared=abstain.prepare(train,seed_posts,scorer,gold)
        def fit_model(rows):
            if args.gate:
                return train_gated(rows,seed_features,corpus,seed,seed_posts,scorer,gold,groups,prepared)
            return train_selector(rows,seed_features)
        predicted=list(range(len(train)))
        for tr,va in splits:
            model=fit_model([train[i] for i in tr])
            for i in va:
                q=train[i]
                predicted[i]=solve(q,baseline[q["qid"]],oof_features,model)
        oof=out/"oof.csv"
        write_checked(oof,predicted,train_file,corpus_file)
        for_report={qid:[{"post_id":pid} for pid in ranks["ids"][:5]] for qid,ranks in train_rank.items()}
        report=evaluate(oof,train_file,train,folds,for_report)
        report["span_changes"]=sum(p["answer"]!=baseline[p["qid"]]["answer"] for p in predicted)
        print(f"Span OOF S={report['S']:.3f}, {report['span_changes']} changed",flush=True)
        model=fit_model(train)
        joblib.dump(model,out/"span.joblib")
    print("Test features",flush=True)
    test_features=build(test,test_rank,corpus,engine,args.top_k)
    baseline={r["qid"]:r for r in _records(args.base_dir/"submission.csv")}
    preds=[solve(q,baseline[q["qid"]],test_features,model) for q in test]
    write_checked(args.output,preds,test_file,corpus_file)
    if not args.model:
        report["elapsed_seconds"]=round(time.perf_counter()-started,1)
        report["revision"]=REVISION
        (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
