"""Grounded, two-endpoint timeline reader for elapsed-day/month archive questions."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import dataclass
from datetime import timedelta
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

import joblib
import numpy as np

import cpu_numeric_2b as num
import cpu_span_2b as qa
import cpu_semantic_2b as semantic
import solution_2b_v2 as base


def supported(row):
    unit=num._norm(row.get("unit") or "")
    text=num._norm(row["question"])
    return row["answer_format"]=="number" and ("месяц" in unit or unit=="дней") and (
        num._is_timeline(text,unit) or bool(re.search(r"сдвину|перенес|разниц.*месяц|дольше|до какого|позднее",text)))


def clauses(q):
    left,right=num._clauses(q)
    if left!=right:
        return left.split("?")[0].strip(", "),right.split("?")[0].strip(", ")
    if re.search(r"сдвину|перенес|позже|позднее",q):
        return "первоначально объявленные сроки "+q,"новая дата переноса или фактическое открытие "+q
    return "первое сообщение, начало "+q,"новое сообщение, завершение "+q


def _positions(ids):
    return {int(pid):i for i,pid in enumerate(ids)}


def _rr(rank):
    return 1/(1+.13*rank) if rank>=0 else 0.


def source_maps(full,seed,dense,e5,role):
    sources=[seed["ids"],[k for _,k in sorted(zip(full["scores"],full["ids"]),reverse=True)]]
    for mixed in (dense,e5):
        sources.append(mixed["ids"])
        if mixed["hop_scores"]:
            sources.append([k for _,k in sorted(((s[role],pid) for pid,s in zip(mixed["ids"],mixed["hop_scores"])),reverse=True)])
        else:
            sources.append(mixed["ids"])
    return list(map(_positions,sources))


def _story_words(posts):
    words=Counter()
    excluded={"области","область","нижнем","новгороде","нижний","новгород","сейчас","будет","глава","года",
              "сообщили","ранее","работы","также","новый","должны","планируют","будут","день","города",
              "нижегородской","нижегородская","нижегородцы","написал","тысяч","нижегородцев"}
    for post in posts:
        first=num._norm(post["text"])[:350]
        words.update({word for word in re.findall(r"[а-я]{5,}",first) if word not in excluded})
    return " ".join(v for v,c in sorted(words.items(),key=lambda p:(-p[1],-len(p[0]),p[0]))[:5] if c>=2)


def _prepare_question(row,source,corpus,engine):
    q=num._norm(row["question"])
    parts=clauses(q)
    output=[]
    cutoff=str(row["asked_at"])[:10]
    story=_story_words([corpus[int(k)] for k in source["seed"]["ids"][:5]])
    for role,part in enumerate(parts):
        maps=source_maps(source["ce"],source["seed"],source["dense"],source["e5"],role)
        candidates=set().union(*(list(m)[:28] for m in maps))
        features={pid:tuple(_rr(m.get(pid,-1)) for m in maps) for pid in candidates}
        def base_score(v):
            return .6*v[0]+.9*v[1]+.3*(v[2]+v[4])+.85*(v[3]+v[5])
        order=sorted(candidates,key=lambda p:(-base_score(features[p]),p))
        shortlist=list(dict.fromkeys(order[:26]+list(maps[3])[:7]+list(maps[5])[:7]+list(maps[1])[:6]))
        if any(corpus[p]["date"]>cutoff for p in shortlist):
            raise ValueError(f"Future endpoint candidate for {row['qid']}")
        search=part + (". Тема: "+story if story else "")
        reranked=engine.encode([(search,corpus[p]["text"]) for p in shortlist],8,False)
        ce={p:float(v) for p,v in zip(shortlist,reranked)}
        ce_pos=_positions(sorted(ce,key=lambda p:(-ce[p],p)))
        priority={p:.72*base_score(features[p])+2.15*_rr(ce_pos[p]) for p in shortlist}
        selected=sorted(shortlist,key=lambda p:(-priority[p],p))[:17]
        output.append([(p,[*features[p],ce[p],ce[p]-max(ce.values()),_rr(ce_pos[p]),priority[p]]) for p in selected])
    return output


def _extra_dates(text,pub):
    week=["понедельник","вторник","сред","четверг","пятниц","суббот","воскресень"]
    for idx,w in enumerate(week):
        for match in re.finditer(r"\bв(?:о)?\s+(?:эту\s+|этот\s+|прошл\w*\s+|ближайш\w*\s+)?"+w+r"\w*",text):
            around=text[max(0,match.start()-60):match.end()+65]
            if num.FUTURE_EVENT.search(around):
                days=(idx-pub.weekday())%7
            else:
                days=-((pub.weekday()-idx)%7)
            yield num.Moment(pub+timedelta(days=days),*match.span(),"weekday")


@dataclass
class Endpoint:
    day: object
    post: int
    features: tuple
    score: float


def _endpoints(row,parts,role,docs,other_docs,corpus,span):
    clause=parts[role];other=parts[1-role]
    mine=num._words(clause); specific=mine-num._words(other)
    months={num._month(m[0]) for m in num.MONTH.finditer(clause)}
    years={int(y) for y in re.findall(r"\b20\d{2}\b",clause)}
    my_events=[e for e in num.EVENTS if e.search(clause) and not e.search(other)]
    their_events=[e for e in num.EVENTS if e.search(other) and not e.search(clause)]
    news=bool(num.NEWS.search(clause))
    explicit=bool(re.search(r"дат[уыой]|объявленн\w+ срок|срока, названного|объявленн\w+ д",clause))
    query=("Когда опубликовали сообщение" if news and not explicit else "Какого числа произошло событие") + " «"+clause+"»?"
    tasks=[]
    pubdays=[]
    for rank,(pid,features) in enumerate(docs):
        d=corpus[pid];pub=num.date.fromisoformat(d["date"]);text=num._norm(d["text"])
        other_score=other_docs.get(pid,0.)
        prefix=f"Дата публикации: {pub.day} {qa.MONTHS[pub.month-1]} {pub.year} года. "
        moments=[(num.Moment(pub,prefix.index(":")+2,prefix.index(" года"),"pub"),True)]
        for m in [*num._dates(text,pub),*_extra_dates(text,pub)]:
            moments.append((num.Moment(m.value,m.start+len(prefix),m.end+len(prefix),m.role),False))
        passage=prefix+text
        tasks.append((pid,features,rank,other_score,pub,text,prefix,passage,moments))
        pubdays.append(pub)
    neural=span.fetch([(query,t[7]) for t in tasks],row["qid"],str(row["asked_at"])[:10])
    by_date=defaultdict(list)
    when=num.date.fromisoformat(row["asked_at"][:10])
    for pid,features,rank,other_score,pub,text,prefix,passage,moments in tasks:
        flags=neural[query,passage]
        docwords=num._words(text)
        topic=num._overlap(mine,docwords)
        own=num._overlap(specific,docwords)
        lead=text[:200]
        here=sum(bool(e.search(lead)) for e in my_events)
        opposite=sum(bool(e.search(lead)) for e in their_events)
        pub_support=sum(_rr(i) for i,d in enumerate(pubdays) if abs((pub-d).days)<=1)
        for moment,is_pub in moments:
            date=moment.value
            if abs((date-pub).days)>730 or date>when and not explicit:
                continue
            logits=qa._span_support(flags,moment.start,moment.end)
            logits=max(-20.,logits if logits is not None else -20.)
            context=lead if is_pub else num._context(passage,moment.start,moment.end)
            local=num._overlap(specific,num._words(context))
            event=sum(bool(e.search(context)) for e in my_events)
            datefit=float(date.month in months)+.5*float(date.year in years)
            age=(date-pub).days
            cues=[float(is_pub),float(is_pub and news),float(not is_pub and explicit),datefit,topic,own,local,
                  here,opposite,event,float(bool(num.FUTURE_EVENT.search(context))),float(bool(num.PAST_EVENT.search(context))),
                  pub_support,math.copysign(math.log1p(abs(age)),age),math.log1p(max(0,(when-date).days)),
                  float(moment.role=="weekday"),float(news),float(explicit),float("перв" in clause)]
            textual=topic*.3+own*.32+local*.37+here*.3+event*.38-opposite*.3+datefit*1.2+min(2,pub_support)*.25
            prior=features[-1]+textual+1.1*(is_pub and news)+.85*(not is_pub and explicit)+.15*logits
            by_date[date].append(Endpoint(date,pid,tuple([*features,*cues,other_score,features[-1]-other_score,
                                                          logits,1/(1+rank),prior]),prior))
    endpoints=[]
    for day,elements in by_date.items():
        best=max(elements,key=lambda e:e.score)
        distinct=len({e.post for e in elements})
        # Multiple reports may refer to the identical planned day.
        endpoints.append(Endpoint(day,best.post,(*best.features,math.log1p(distinct)),best.score+.32*math.log1p(distinct)))
    return sorted(endpoints,key=lambda e:(-e.score,e.day))[:32]


def make_pairs(row,prepared,corpus,span):
    parts=clauses(num._norm(row["question"]))
    opts=[]
    for role in (0,1):
        other={pid:features[-1] for pid,features in prepared[1-role]}
        opts.append(_endpoints(row,parts,role,prepared[role],other,corpus,span))
    rows=[]
    for ia,a in enumerate(opts[0]):
        for ib,b in enumerate(opts[1]):
            if a.day>=b.day:
                continue
            n=(b.day-a.day).days
            if "месяц" in row["unit"]:
                n=(b.day.year-a.day.year)*12+b.day.month-a.day.month-int("полн" in (row["question"]+row["unit"]) and b.day.day<a.day.day)
            if n<0:
                continue
            prior=(a.score+b.score)/2
            v=[prior,min(a.score,b.score),1/(1+ia),1/(1+ib),float(a.post==b.post),math.log1p(n),
               float("месяц" in row["unit"]),*a.features,*b.features]
            rows.append((str(n),a.post,b.post,v))
    return rows


def train_pairs(rows,pairs):
    from lightgbm import LGBMRanker
    x,y,groups=[],[],[]
    for q in rows:
        examples=pairs[q["qid"]]
        gold={int(k) for k in q["evidence"].split()}
        def label(item):
            v,p1,p2,_=item
            if not num._match(v,q["answer"],"number"):
                return 0
            return 1+int(p1 in gold)+int(p2 in gold)
        targets=[label(item) for item in examples]
        if not targets or not max(targets):
            continue
        chosen=sorted(range(len(examples)),key=lambda i:-examples[i][3][0])
        hard=[i for i in chosen if targets[i]==0]
        keep=[i for i in chosen if targets[i]>0]+hard[:80]+hard[80::5]
        groups.append(len(keep));x.extend(examples[i][3] for i in keep);y.extend(targets[i] for i in keep)
    if not groups:
        return None
    model=LGBMRanker(n_estimators=95,num_leaves=9,learning_rate=.05,min_child_samples=24,
                     reg_lambda=4,lambdarank_truncation_level=14,n_jobs=2,verbosity=-1,
                     random_state=31,deterministic=True,force_col_wise=True)
    model.fit(np.asarray(x,np.float32),y,group=groups)
    return model


def pair_answer(model,samples):
    if not samples or model is None:
        return None
    x=np.asarray([item[3] for item in samples],np.float32)
    logits=model.booster_.predict(x,num_threads=1)
    logits=logits+.40*(x[:,0]-x[:,0].max())
    by_answer=defaultdict(list)
    for i,item in enumerate(samples):
        by_answer[item[0]].append(i)
    answers=[]
    for v,ix in by_answer.items():
        ordered=sorted(ix,key=lambda i:-logits[i])
        top=ordered[0]
        support=sum(math.exp(max(-15,logits[i]-logits[top])) for i in ordered[:5])
        score=logits[top]+.34*math.log(support)
        answers.append((score,top,v))
    answers.sort(reverse=True)
    s,idx,v=answers[0]
    sample=samples[idx]
    features=[s,s-answers[1][0] if len(answers)>1 else 1,logits[idx],x[idx,0],x[idx,1],
              x[idx,2],x[idx,3],x[idx,4],x[idx,6],math.log1p(len(by_answer)),
              math.log1p(len(by_answer[v]))]
    return v,list(dict.fromkeys(sample[1:3])),features


def make_record(row,base_row,result,keep_anchors):
    if result is None:
        return base_row.copy()
    answer,anchors,_=result
    old=[int(k) for k in base_row["evidence"].split()]
    ids=list(dict.fromkeys((anchors if keep_anchors else [])+old))[:5]
    return dict(qid=row["qid"],answer=answer,evidence=" ".join(map(str,ids)))


def _utility(scorer,gold,record):
    return scorer.score_item(gold,dict(answer=record["answer"],evidence=record["evidence"].split()))["s"]


def action_features(q,base_row,result):
    ans,anchors,f=result
    evidence={int(k) for k in base_row["evidence"].split()}
    return [*f,float(ans==base_row["answer"]),float(base_row["answer"]=="skip"),
            len(set(anchors)&evidence)/max(1,len(anchors)),float("между" in q["question"]),
            float("перв" in q["question"]),float("позже" in q["question"])]


def train_with_policy(rows,pairs,independent,groups,scorer,gold):
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    xs,ys,ws=[],[],[]
    for tr,va in GroupKFold(5).split(rows,groups=[groups[q["qid"]] for q in rows]):
        model=train_pairs([rows[i] for i in tr],pairs)
        for i in va:
            q=rows[i];qid=q["qid"];old=independent[qid]
            result=pair_answer(model,pairs[qid])
            if result is None:
                continue
            xs.append(action_features(q,old,result))
            s0=_utility(scorer,gold[qid],old)
            ys.append([_utility(scorer,gold[qid],make_record(q,old,result,anchor))-s0
                       for anchor in (False,True)])
            ws.append(gold[qid]["_w"])
    policy=None
    if len(xs)>=10:
        policy=make_pipeline(StandardScaler(),Ridge(alpha=65))
        w=np.asarray(ws)
        policy.fit(xs,ys,ridge__sample_weight=w/w.mean())
    return dict(pairs=train_pairs(rows,pairs),policy=policy)


def solve(q,incumbent,pairs,models):
    result=pair_answer(models["pairs"],pairs[q["qid"]])
    raw=make_record(q,incumbent,result,True)
    if result is None or models["policy"] is None:
        return incumbent,raw
    gains=models["policy"].predict([action_features(q,incumbent,result)])[0]
    action=int(np.argmax([0.,*gains]))
    return (make_record(q,incumbent,result,action==2) if action else incumbent),raw


def main():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from run_cpu_2b import write_checked

    p=argparse.ArgumentParser(description=__doc__)
    for k in ("corpus","train","test","groups","model"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--base-dir",type=Path,required=True)
    p.add_argument("--rank-dir",type=Path,required=True)
    p.add_argument("--seed-dir",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path,default=base.ROOT/"output_2b_final/timeline_cache")
    p.add_argument("--weights",type=Path,default=base.ROOT/"weights_2b")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=3)
    args=p.parse_args();started=time.perf_counter()
    dest=args.output.parent;dest.mkdir(parents=True,exist_ok=True);args.cache_dir.mkdir(parents=True,exist_ok=True)
    qfiles={k:base.find_file(base.ROOT,k+".csv",getattr(args,k)) for k in (["test"] if args.model else ["train","test"])}
    corpus_path=base.find_file(base.ROOT,"corpus.parquet",args.corpus)
    questions={k:qa._records(path) for k,path in qfiles.items()}
    timeline={k:[q for q in rows if supported(q)] for k,rows in questions.items()}
    inputs={}
    for split,rows in timeline.items():
        sources={k:semantic.read_ranks((args.seed_dir if k=="seed" else args.rank_dir)/f"{k}_{split}.jsonl")
                 for k in ("seed","ce","dense","e5")}
        for q in rows:
            inputs[q["qid"]]={k:v[q["qid"]] for k,v in sources.items()}
    wanted=pa.array(sorted({int(pid) for source in inputs.values() for mixed in source.values() for pid in mixed["ids"]}))
    corpus={}
    for batch in pq.ParquetFile(corpus_path).iter_batches(batch_size=4096,columns=["post_id","text","date"]):
        for post in batch.filter(pc.is_in(batch.column("post_id"),value_set=wanted)).to_pylist():
            post["date"]=str(post["date"])[:10]
            corpus[int(post["post_id"])]=post
    rows=sum(timeline.values(),[])
    version=hashlib.sha256(Path(__file__).read_bytes()+Path(qa.__file__).read_bytes()+Path(num.__file__).read_bytes()).hexdigest()
    checkpoint={}
    for q in rows:
        qid=q["qid"]
        source=inputs[qid]
        text_hash=hashlib.sha256()
        for pid in sorted({int(pid) for r in source.values() for pid in r["ids"]}):
            post=corpus[pid]
            if post["date"]>q["asked_at"]:
                raise ValueError(f"Future input for {qid}")
            text_hash.update((str(pid)+post["date"]+post["text"]).encode())
        signature=json.dumps([version,q["question"],qid,q["asked_at"],q["unit"],source,text_hash.hexdigest()],sort_keys=True)
        checkpoint[qid]=args.cache_dir/(hashlib.sha256(signature.encode()).hexdigest()+".joblib")
    prepared={}
    engine=semantic.Engine(args.weights/"cpu_ce",args.threads,192)
    for i,q in enumerate(rows):
        qid=q["qid"]
        if not checkpoint[qid].exists():
            prepared[qid]=_prepare_question(q,inputs[qid],corpus,engine)
        if (i+1)%20==0:
            print(f"Timeline event retrieval {i+1}/{len(rows)}",flush=True)
    del engine;gc.collect()
    span=qa.SpanEngine(args.weights/"cpu_span",args.cache_dir/"qa.sqlite",args.threads,256,6)
    pairs={}
    for i,q in enumerate(rows):
        qid=q["qid"];path=checkpoint[qid]
        if path.exists():
            pairs[qid]=joblib.load(path)
        else:
            pairs[qid]=make_pairs(q,prepared[qid],corpus,span)
            tmp=path.with_suffix(".part")
            joblib.dump(pairs[qid],tmp,compress=2);tmp.replace(path)
        if (i+1)%20==0:
            print(f"Grounded timeline date pairs {i+1}/{len(rows)}",flush=True)
    del span;gc.collect()
    report={}
    if args.model:
        models=joblib.load(args.model)
    else:
        train=questions["train"]
        base_rows=qa._records(args.base_dir/"oof.csv")
        incumbent={r["qid"]:r for r in base_rows}
        group_path=base.find_file(base.ROOT,"train_groups.csv",args.groups)
        mapping={r["qid"]:r["group"] for r in qa._records(group_path)}
        folds,splits=base.group_folds(base.read_questions(qfiles["train"]),group_path)
        scorer=base.module_from_file(base.find_file(base.ROOT,"score.py"))
        gold={r["id"]:r for r in scorer.load_gold([str(qfiles["train"])])}
        independent={}
        for q in timeline["train"]:
            ids=inputs[q["qid"]]["seed"]["ids"][:60]
            ans=num.predict(q,[corpus[int(k)] for k in ids])
            independent[q["qid"]]=dict(qid=q["qid"],answer=ans,evidence=" ".join(map(str,ids[:5])))
        pred=dict(incumbent);raw=dict(incumbent)
        for f,(tr,va) in enumerate(splits):
            models=train_with_policy([train[i] for i in tr if supported(train[i])],pairs,independent,mapping,scorer,gold)
            for i in va:
                q=train[i];qid=q["qid"]
                if qid in pairs:
                    pred[qid],raw[qid]=solve(q,incumbent[qid],pairs,models)
            print(f"Timeline fold {f+1}/5",flush=True)
        oof=dest/"oof.csv"
        write_checked(oof,[pred[q["qid"]] for q in train],qfiles["train"],corpus_path)
        def result(table):
            scored=[scorer.score_item(gold[q["qid"]],dict(answer=table[q["qid"]]["answer"],
                      evidence=table[q["qid"]]["evidence"].split())) for q in train]
            ts=[r for r in scored if r["qid"] in pairs]
            return dict(S=scorer.total(scored),timeline_n=len(ts),timeline_correct=sum(r["a"]>0 for r in ts),
                        timeline_S=scorer.total(ts),folds=[scorer.total([r for i,r in enumerate(scored) if folds[i]==f]) for f in range(5)])
        report={k:result(v) for k,v in (("base",incumbent),("ungated",raw),("nested",pred))}
        report["timeline_answer_coverage"]=sum(any(num._match(s[0],q["answer"],"number") for s in pairs[q["qid"]]) for q in timeline["train"])
        print(json.dumps(report),flush=True)
        models=train_with_policy(timeline["train"],pairs,independent,mapping,scorer,gold)
        joblib.dump(models,dest/"timeline.joblib")
    incumbent={r["qid"]:r for r in qa._records(args.base_dir/"submission.csv")}
    for q in timeline["test"]:
        incumbent[q["qid"]],_=solve(q,incumbent[q["qid"]],pairs,models)
    write_checked(args.output,[incumbent[q["qid"]] for q in questions["test"]],qfiles["test"],corpus_path)
    report["elapsed_seconds"]=round(time.perf_counter()-started,1)
    (dest/("inference.json" if args.model else "report.json")).write_text(json.dumps(report,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
