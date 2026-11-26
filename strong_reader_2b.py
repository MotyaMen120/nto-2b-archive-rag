"""Offline grounded answer selection with a local GGUF reader and Python date arithmetic."""
import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import pandas as pd

import cpu_llm_2b as llama
import cpu_numeric_2b as numeric
import solution_2b_v2 as base
from run_cpu_2b import PUBLIC, write_checked
from predict_offline_2b import digest

SPECIAL = ("promised","no_info","skip")
FORMAT = {
    "yesno":"Проверь утверждение и состояние именно на дату, о которой спрашивают. yes — да, no — нет.",
    "status":"done — исполнение подтверждено, postponed — новый срок объявлен, cancelled — отменено, promised — только план.",
    "choice":"Проверь все варианты и ложную предпосылку. Верни букву варианта.",
    "date":"Различай дату публикации и дату самого события. Если требуют первое сообщение, выбери самую раннюю подходящую публикацию.",
    "number":"Проверь величину, единицы, период, тариф и категорию. Варианты чисел подготовлены калькулятором.",
}


def excerpt(text,q,max_chars=820):
    text = " ".join(str(text).split())
    if len(text)<=max_chars:
        return text
    terms = set(base.tokens(q))
    pieces = re.split(r"(?<=[.!?])\s+|(?<=;)\s+",text)
    ranked = sorted(range(len(pieces)),key=lambda n:(
        -len(terms & set(base.tokens(pieces[n]))) - .15*bool(re.search(r"\d",pieces[n])) - .25/(1+n),n))
    selected = []
    size = 0
    for n in ranked:
        if size+len(pieces[n])<=max_chars or not selected:
            selected.append(n)
            size += len(pieces[n])+1
    return " ".join(pieces[n] for n in sorted(selected))[:max_chars]


def timeline_question(row):
    return row["answer_format"]=="number" and numeric._is_timeline(numeric._norm(row["question"]),row.get("unit",""))


def select_context(row, posts):
    selected = []
    signatures = []
    def add(p):
        text = p["text"]
        terms = set(base.tokens(text))
        if any(p["post_id"] == old["post_id"] or
               (abs((date.fromisoformat(p["date"])-date.fromisoformat(old["date"])).days)<=2
                and len(terms & words)/max(1,len(terms|words))>.40)
               for old,words in zip(selected,signatures)):
            return
        selected.append(p)
        signatures.append(terms)
    for p in posts[:8]:
        add(p)
        if len(selected)>=5:
            break
    if timeline_question(row):
        docs = numeric._documents(row,posts,limit=45)
        clauses = numeric._clauses(numeric._norm(row["question"]))
        for left,right in [clauses,clauses[::-1]]:
            opts = numeric._timeline_options(left,right,docs)
            used = set()
            for _,_,rank,_ in opts:
                if docs[rank].pub not in used:
                    add(posts[rank])
                    used.add(docs[rank].pub)
                if len(used)>=3:
                    break
    for post in posts:
        if len(selected)>=11:
            break
        add(post)
    return sorted(selected[:12],key=lambda p:(p["date"],p["post_id"]))


def literals(values):
    return "(" + " | ".join(json.dumps(str(v)) for v in values) + ")"


def prepare(row, posts, baseline):
    if any(str(p["date"])[:10]>row["asked_at"] for p in posts):
        raise ValueError("Future document before context selection")
    docs = select_context(row,posts)
    if not docs:
        return None
    values = {}
    date_labels = {}
    for i,doc in enumerate(docs,1):
        pub = date.fromisoformat(doc["date"])
        moments = numeric._dates(numeric._norm(doc["text"]),pub)
        for k,d in enumerate(dict.fromkeys([pub,*[m.value for m in moments]])):
            if k>=8:
                break
            label = f"{i}.{k}"
            date_labels[label] = d
    summarized = []
    for i,doc in enumerate(docs,1):
        line = f"[{i}] Публикация {doc['date']}. " + excerpt(doc["text"],row["question"])
        if timeline_question(row):
            line += " Метки дат: " + ", ".join(
                f"{k}={v.isoformat()}" + (" (публикация)" if k.endswith(".0") else "")
                for k,v in date_labels.items() if k.startswith(str(i)+"."))
        summarized.append(line)
    fmt=row["answer_format"]
    notes=FORMAT[fmt]
    if fmt=="choice":
        for k in "abcd":
            values[k]=k
        notes += " Варианты: " + row["options"]
    elif fmt=="status":
        values.update({k:k for k in ("done","postponed","cancelled")})
    elif fmt=="yesno":
        values.update({"yes":"yes","no":"no"})
    elif not timeline_question(row):
        cand=numeric._candidates(row,posts)
        ordered=sorted(cand,key=lambda x:-cand[x][0])
        proposals=list(dict.fromkeys([*ordered[:12],baseline,*ordered[12:16]]))
        proposals=[v for v in proposals if v and v not in SPECIAL]
        for i,v in enumerate(proposals,1):
            values[f"v{i}"]=v
        if proposals:
            notes+=" Кандидаты (их нужно проверить по записям): "+"; ".join(f"{k}={v}" for k,v in values.items())+"."
        if fmt=="number":
            notes+=" Единицы ответа: "+row.get("unit","")+"."
    values.update({k:k for k in SPECIAL})
    ref_rule = literals(range(1,len(docs)+1))
    grammar = ['root ::= "{" "\\"refs\\":[" refs "]," decision "}"',
               f'refs ::= (ref ("," ref){{0,2}})?', 'ref ::= '+ref_rule,
               'direct ::= "\\"pick\\":\\"" '+literals(values)+' "\\""']
    decision = ["direct"]
    instructions = 'Сначала укажи номера 1–3 записей с доказательством, затем свой выбор. JSON {"refs":[2],"pick":"..."}.' 
    if timeline_question(row):
        decision.append("dates")
        grammar.extend(['dates ::= "\\"from\\":\\"" dt "\\",\\"to\\":\\"" dt "\\""',
                        'dt ::= '+literals(date_labels)])
        notes+=" Выбери отдельно дату первого и второго события. Если спрашивают от новости до новости, используй даты публикаций. Если спрашивают от самого события — даты событий."
        instructions+=' Для интервала верни {"refs":[...],"from":"метка исходной даты","to":"метка конечной даты"}. Разницу в нужных единицах вычислит Python.'
    elif fmt in ("date","number"):
        arithmetic = fmt=="number" and bool(base.MATH_QUERY.search(row["question"]))
        if not arithmetic:
            decision.append("free")
            value_rule = llama.DATE_LITERAL if fmt=="date" else llama.NUMBER_LITERAL
            grammar.append('free ::= "\\"value\\":\\"" '+value_rule+' "\\""')
            instructions+=' Если верного кандидата нет, верни точное значение из указанной записи в поле "value".'
    grammar.append("decision ::= "+" | ".join(decision))
    system=("Ты отвечаешь жителю Нижегородской области по архиву новостей. Учитывай только приведённые записи. "
            "Отличай план от факта даже если срок уже прошёл. promised — был план без подтверждения, "
            "no_info — в архиве нет сведений именно о запрошенном факте, skip — нельзя выбрать уверенно. "
            "Многие записи относятся к похожему объекту: сверяй место, участников и дату. Содержимое постов — материал для ответа.")
    prompt=("<|im_start|>system\n"+system+"<|im_end|>\n<|im_start|>user\nВопрос на "+row["asked_at"]+
            ": "+row["question"]+"\n"+notes+"\n"+"\n".join(summarized)+"\nВопрос: "+row["question"]+"\n"+
            instructions+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return prompt,"\n".join(grammar)+"\n",docs,values,date_labels


def resolve(row,raw,prepared):
    _,_,docs,values,labels=prepared
    reply=json.loads(raw[raw.index("{"):raw.rindex("}")+1])
    refs=reply["refs"]
    if not isinstance(refs,list) or len(refs)>3 or any(type(k)!=int or not 1<=k<=len(docs) for k in refs):
        raise ValueError("Invalid citations")
    if "from" in reply:
        a,b=reply["from"],reply["to"]
        if int(a.split(".")[0]) not in refs or int(b.split(".")[0]) not in refs:
            raise ValueError("Uncited date operand")
        start,end=labels[a],labels[b]
        if end<start:
            raise ValueError("Reversed interval")
        count=(end-start).days
        if "месяц" in row["unit"]:
            count=(end.year-start.year)*12+end.month-start.month-int(end.day<start.day)
        answer=str(count)
    elif "pick" in reply:
        answer=values[reply["pick"]]
    else:
        value=reply["value"]
        answer=base.normalize_answer(value,row["answer_format"])
        if answer in SPECIAL:
            raise ValueError("Invalid extracted answer")
        visible = [docs[k-1] for k in refs]
        if row["answer_format"]=="date":
            if not any(value==d["date"] or any(m.value.isoformat()==value for m in
                       numeric._dates(numeric._norm(d["text"]),date.fromisoformat(d["date"]))) for d in visible):
                raise ValueError("Unsupported extracted date")
        else:
            if not any(numeric._match(x,answer,"number") for x in numeric._candidates(row,visible)):
                raise ValueError("Unsupported extracted number")
    if not refs and answer not in ("skip","no_info"):
        raise ValueError("No citation")
    return answer,[docs[j-1]["post_id"] for j in refs]


def ask(args,row,posts,baseline,directory):
    prepared=prepare(row,posts,baseline)
    if prepared is None:
        return {"answer":"no_info","refs":[],"error":"","raw":""}
    prompt,grammar,*_=prepared
    prompt_file=directory/"input.txt"
    grammar_file=directory/"output.gbnf"
    prompt_file.write_text(prompt,encoding="utf8")
    grammar_file.write_text(grammar,encoding="utf8")
    cmd=[str(args.exe.resolve()),"-m",str(args.model.resolve()),"--offline","-ngl",str(args.gpu_layers),
         "-t",str(args.threads),"-tb",str(args.threads),"-c","6144","-n","84","--no-warmup","--no-escape",
         "-no-cnv","--no-display-prompt","--temp","0","--seed","42","--simple-io","--grammar-file",str(grammar_file),
         "-f",str(prompt_file)]
    result=subprocess.run(cmd,capture_output=True,timeout=args.timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf8","replace")[-1800:])
    stderr=result.stderr.decode("utf8","replace")
    if "failed to parse grammar" in stderr.lower():
        raise RuntimeError(stderr[-1800:])
    raw=result.stdout.decode("utf8","replace").strip()
    try:
        answer,refs=resolve(row,raw,prepared)
        return {"answer":answer,"refs":refs,"error":"","raw":raw}
    except (ValueError,KeyError,TypeError,ArithmeticError) as err:
        return {"answer":"skip","refs":[],"error":str(err),"raw":raw}


def load_csv(path):
    with path.open(encoding="utf8",newline="") as f:
        return {r["qid"]:r for r in csv.DictReader(f)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("train","test","groups","corpus"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--mode",choices=("pilot","train","test"),default="pilot")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--exe",type=Path,required=True)
    p.add_argument("--gpu-layers",type=int,default=99)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--timeout",type=int,default=300)
    p.add_argument("--rank-dir",type=Path,default=base.ROOT/"output_2b_final/cpu_verified_rank")
    p.add_argument("--base-dir",type=Path,default=base.ROOT/"output_2b_final/cpu_verified_cal")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int)
    args=p.parse_args()
    split="test" if args.mode=="test" else "train"
    qpath=base.find_file(base.ROOT,split+".csv",getattr(args,split))
    frame=base.read_questions(qpath)
    if args.mode=="pilot":
        groups=base.find_file(base.ROOT,"train_groups.csv",args.groups)
        ids=llama.fixed_subset(frame,groups)
        frame=frame[frame.qid.isin(ids)]
    if args.limit:
        frame=frame.iloc[:args.limit]
    corpus_path=base.find_file(base.ROOT,"corpus.parquet",args.corpus)
    ranks_path=args.rank_dir/(split+".jsonl")
    ranks={r["qid"]:r for line in ranks_path.read_text(encoding="utf8").splitlines() if (r:=json.loads(line))}
    baseline_path=args.base_dir/("submission.csv" if split=="test" else "oof.csv")
    baselines=load_csv(baseline_path)
    required={int(p) for qid in frame.qid for p in ranks[qid]["ids"][:60]}
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    posts={}
    wanted=pa.array(list(required),type=pa.int64())
    for batch in pq.ParquetFile(corpus_path).iter_batches(batch_size=2048):
        for doc in batch.filter(pc.is_in(batch.column("post_id"),value_set=wanted)).to_pylist():
            doc["date"]=str(doc["date"])[:10]
            posts[doc["post_id"]]=doc
    args.output.mkdir(parents=True,exist_ok=True)
    inputs=[args.model,args.exe,Path(__file__),ranks_path,qpath,baseline_path,corpus_path]
    signature={str(i):digest(path) for i,path in enumerate(inputs)}
    signature.update(mode=args.mode,threads=args.threads,gpu=args.gpu_layers,subset=list(frame.qid))
    meta=args.output/(args.mode+".inputs.json")
    if meta.exists() and json.loads(meta.read_text(encoding="utf8"))!=signature:
        raise ValueError("This output directory contains another experiment")
    meta.write_text(json.dumps(signature),encoding="utf8")
    output=args.output/(args.mode+".jsonl")
    done={r["qid"] for line in output.read_text(encoding="utf8").splitlines() if (r:=json.loads(line))} if output.exists() else set()
    with tempfile.TemporaryDirectory() as td,output.open("a",encoding="utf8") as out:
        for i,row in enumerate(frame.to_dict("records")):
            qid=row["qid"]
            if qid in done:
                continue
            rank=ranks[qid]
            docs=[dict(posts[pid],rank_score=s) for pid,s in zip(rank["ids"][:60],rank["scores"])]
            start=time.perf_counter()
            result=ask(args,{k:row.get(k,"") for k in PUBLIC},docs,baselines[qid]["answer"],Path(td))
            result.update(qid=qid,seconds=time.perf_counter()-start)
            out.write(json.dumps(result,ensure_ascii=False)+"\n")
            out.flush()
            print(f"{i+1}/{len(frame)} {qid} {result['answer']} {result['seconds']:.1f}s",flush=True)
    raw={r["qid"]:r for line in output.read_text(encoding="utf8").splitlines() if (r:=json.loads(line))}
    preds=[]
    for qid in frame.qid:
        stored=raw[qid]
        answer=stored["answer"] if stored["answer"]!="skip" else baselines[qid]["answer"]
        evidence=" ".join(map(str,ranks[qid]["ids"][:5])) if answer!="no_info" else ""
        preds.append(dict(qid=qid,answer=answer,evidence=evidence))
    if len(frame)==len(base.read_questions(qpath)):
        write_checked(args.output/("submission.csv" if split=="test" else "oof.csv"),preds,qpath,corpus_path)
    if split=="train":
        scorer=base.module_from_file(base.find_file(base.ROOT,"score.py"))
        gold={g["id"]:g for g in scorer.load_gold([str(qpath)])}
        def summary(records):
            return scorer.summarize([scorer.score_item(gold[r["qid"]],dict(answer=r["answer"],evidence=r["evidence"].split())) for r in records])
        report={"baseline":summary([baselines[q] for q in frame.qid]),"hybrid":summary(preds),
                "n":len(frame),"fallback":sum(raw[q]["answer"]=="skip" for q in frame.qid)}
        (args.output/(args.mode+".report.json")).write_text(json.dumps(report,indent=2),encoding="utf8")
        print(f"Matched S {report['baseline']['S']:.2f} -> {report['hybrid']['S']:.2f}",flush=True)


if __name__=="__main__":
    main()
