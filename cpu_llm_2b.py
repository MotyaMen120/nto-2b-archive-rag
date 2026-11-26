"""Standalone llama.cpp reader and grouped, deterministic CPU pilot for NTO 2B."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import inspect
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time

import pandas as pd

import solution_2b_v2 as base


def adapt_ollama(source, output):
    """Repack the Ollama Qwen3.5 multimodal blob as a llama.cpp text-only GGUF.

    Keep text tensor bytes and order. Remove the vision encoder and pad the MRoPE
    metadata with a fourth zero axis, as required by llama.cpp's text executor.
    """
    def uint64(f):
        return struct.unpack("<Q", f.read(8))[0]
    def string(f):
        return f.read(uint64(f))
    sizes = (1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8)
    with source.open("rb") as f:
        signature = f.read(8)
        if signature != b"GGUF\x03\x00\x00\x00":
            raise ValueError("Expected GGUF v3.")
        tensors, fields = uint64(f), uint64(f)
        alignment = 32
        updates = []
        for _ in range(fields):
            key = string(f)
            typ = struct.unpack("<I", f.read(4))[0]
            if typ == 9:
                elem = struct.unpack("<I", f.read(4))[0]
                count = uint64(f)
                if key == b"qwen35.rope.dimension_sections" and count == 3 and elem == 5:
                    at = f.tell()
                    raw = f.read(12)
                    updates.append((at - 8, at + 12, struct.pack("<Q", 4) + raw + bytes(4)))
                elif key == b"qwen35.attention.head_count_kv" and elem in (4, 5):
                    at = f.tell()
                    heads = {h for h in struct.unpack(f"<{count}I", f.read(4 * count)) if h}
                    if len(heads) != 1:
                        raise ValueError("Expected uniform KV head count on dense attention layers.")
                    updates.append((at - 16, f.tell(), struct.pack("<II", 4, heads.pop())))
                elif elem == 8:
                    for _ in range(count):
                        string(f)
                else:
                    f.seek(count * sizes[elem], 1)
            elif typ == 8:
                string(f)
            elif key == b"general.alignment":
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                f.seek(sizes[typ], 1)
        table = f.tell()
        index = []
        for _ in range(tensors):
            name = string(f)
            nd = struct.unpack("<I", f.read(4))[0]
            info = struct.pack("<I", nd) + f.read(nd * 8 + 4)
            index.append((name, info, uint64(f)))
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        total = source.stat().st_size
        if not updates or not any(t[0].startswith(b"v.") for t in index):
            raise ValueError("Provide a llama.cpp GGUF directly with --model or an Ollama Qwen3.5 multimodal blob.")
        f.seek(0)
        metadata = bytearray(f.read(table))
        for start, stop, data in reversed(updates):
            metadata[start:stop] = data
        offsets = sorted({t[2] for t in index} | {total - data_start})
        length = dict(zip(offsets[:-1], (b - a for a, b in zip(offsets, offsets[1:]))))
        text = [t for t in index if t[0].startswith((b"blk.", b"token_embd.", b"output.", b"output_norm."))]
        metadata[8:16] = struct.pack("<Q", len(text))
        position = 0
        for name, info, start in text:
            if name.endswith(b".ssm_dt"):
                name += b".bias"
            metadata += struct.pack("<Q", len(name)) + name + info + struct.pack("<Q", position)
            position += length[start]
        output.parent.mkdir(exist_ok=True, parents=True)
        temp = output.with_suffix(".tmp")
        with temp.open("wb") as dest:
            dest.write(metadata)
            dest.write(bytes((-len(metadata)) % alignment))
            for _, _, start in text:
                f.seek(data_start + start)
                remaining = length[start]
                while remaining:
                    chunk = f.read(min(8 << 20, remaining))
                    if not chunk:
                        raise EOFError("Incomplete GGUF")
                    dest.write(chunk)
                    remaining -= len(chunk)
        temp.replace(output)
    print(f"Exported {len(text)} text tensors to {output}", flush=True)


SPECIAL = ('promised', 'no_info', 'skip')
NUMBER_LITERAL = r'("-"? [0-9]+ ("." [0-9]+)?)'
DATE_LITERAL = r'([0-9]{4} "-" [0-9]{2} "-" [0-9]{2})'
MATH_OPS = ("days_diff", "full_months_diff", "sub", "mul", "div", "sum", "pct_change", "count")


def context(q, ranks, posts, max_posts=6, excerpt_chars=600):
    selected, signatures = [], []
    for pid in ranks['ids']:
        row = posts.loc[pid]
        when = str(row.date)[:10]
        if when > q['asked_at']:
            raise ValueError('Future post in input ranks')
        text = ' '.join(str(row.text).split())
        words = set(base.tokens(text))
        if any(len(words & terms) / max(1, len(words | terms)) > .45 and
               abs((date.fromisoformat(when) - date.fromisoformat(d)).days) <= 2 for terms, d in signatures):
            continue
        signatures.append((words, when))
        selected.append({'post_id': int(pid), 'date': when, 'text': text[:excerpt_chars]})
        if len(selected) >= max_posts:
            break
    return selected


def alternation(values):
    return '(' + ' | '.join(json.dumps(x) for x in values) + ')'


def prompt_grammar(q, docs):
    fmt = q["answer_format"]
    arithmetic = fmt == 'number' and bool(base.MATH_QUERY.search(q["question"]))
    enums = {"status":["done", "cancelled", "postponed"], "yesno":["yes", "no"], "choice":list("abcd")}
    fmt_rule = alternation(enums.get(fmt, []) + list(SPECIAL))
    if fmt == 'date':
        fmt_rule += ' | ' + DATE_LITERAL
    if fmt == 'number' and not arithmetic:
        fmt_rule += ' | ' + NUMBER_LITERAL
    n = len(docs)
    grammar = ['root ::= direct' + (' | calculation' if arithmetic else ''),
               'direct ::= ' + json.dumps('{"answer":"') + ' ans ' + json.dumps('","refs":') + ' refs "}"',
               'ans ::= ' + fmt_rule,
               f'refs ::= "[" (index ("," index){{0,{min(n-1,4)}}})? "]"',
               'index ::= ' + alternation([str(i+1) for i in range(n)])]
    extra = ''
    if arithmetic:
        grammar.append('calculation ::= ' + json.dumps('{"op":"') + ' op ' + json.dumps('","a":"') + ' value ' +
                       json.dumps('","b":"') + ' value ' + json.dumps('","refs":') + ' refs "}"')
        grammar.extend(['value ::= ' + DATE_LITERAL + ' | ' + NUMBER_LITERAL, 'op ::= ' + alternation(MATH_OPS)])
        extra = (' Для расчета верни {"op":"...","a":"...","b":"...","refs":[номер_источника_a,номер_источника_b]}. '
                 'Для разницы дат укажи начальную и конечную даты YYYY-MM-DD, op=days_diff или full_months_diff. '
                 'Числа из текста: sub это a-b, sum это a+b, mul, div, pct_change. Python вычислит ответ.')
    prompt = ('<|im_start|>system\nТы отвечаешь по городскому архиву. Используй только подходящие по смыслу факты. '
              'План без подтверждения выполнения — promised. Нет сведений по теме — no_info. '
              'Если спрашивают о прошлом, проверь состояние на ту дату. Для «впервые сообщили» нужна дата публикации. '
              'Укажи ответ и номера записей, в которых он подтвержден. JSON без пояснений.<|im_end|>\n'
              '<|im_start|>user\n' + '\n'.join(f"{i}. [{d['date']}] {d['text']}" for i, d in enumerate(docs, 1)) +
              '\n\nНа дату ' + q["asked_at"] + '. Вопрос: ' + q["question"] + '\n' +
              ("Варианты: " + q["options"] + '\n' if q.get("options") else '') +
              {"yesno":"Ответ yes или no.", "choice":"Ответ буквой a,b,c,d. Проверь ложные предпосылки.",
               "status":"Статус done, promised, cancelled или postponed.",
               "number":"Ответ числом в единицах " + str(q.get("unit","")) + ".",
               "date":"Ответ датой YYYY-MM-DD."}[fmt] +
              ' Формат {"answer":"...","refs":[1]}. refs — номера опорных записей в списке.' + extra +
              '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    return prompt, '\n'.join(grammar) + '\n'


def parse(raw, docs, q):
    result = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
    refs = result["refs"]
    if not isinstance(refs, list) or len(refs) > 5 or any(type(i) is not int or not 1 <= i <= len(docs) for i in refs):
        raise ValueError('Invalid citation indices')
    cited = [docs[i-1]['post_id'] for i in refs]
    if 'op' in result:
        if len(refs) < 2:
            raise ValueError('Missing operand citations')
        op = result['op']
        if op == 'count':
            raise ValueError('Event count requires distinct grounded event descriptions')
        values = []
        for name, i in zip(('a','b'), refs):
            value = str(result[name]); doc = docs[i-1]
            source = {'post_id':doc['post_id'],'value':value,'quote':doc['text']}
            if op in ('days_diff', 'full_months_diff') and value == doc['date']:
                source['source'] = 'publication_date'
            values.append(source)
        expr = {'op':op, 'a':values[0], 'b':values[1], 'values':values}
        answer = base.calculate(expr, {d['post_id']:d for d in docs}, q)
    else:
        answer = base.normalize_answer(result['answer'], q['answer_format'])
        if q['answer_format'] == 'number' and base.MATH_QUERY.search(q["question"]) and answer not in SPECIAL:
            raise ValueError("Calculation required")
    if answer not in ('skip','no_info') and not cited:
        raise ValueError('Missing citation')
    return answer, list(dict.fromkeys(cited))


def infer(exe, model, threads, gpu_layers, directory, q, docs, ctx_size=3584):
    if not docs:
        return {"answer":"no_info", "cited":[], "raw":"","seconds":0}
    prompt, grammar = prompt_grammar(q, docs)
    fp, gp = directory/'prompt.txt', directory/'grammar.gbnf'
    fp.write_text(prompt,encoding='utf8');gp.write_text(grammar,encoding='utf8')
    started = time.perf_counter()
    p = subprocess.run([str(exe), '-m', str(model), '--offline', '-ngl', str(gpu_layers), '-t',str(threads),'-tb',str(threads),
                        '-c',str(ctx_size),'-n','96','--no-escape','--no-warmup','-no-cnv','--no-display-prompt','--temp','0',
                        '--seed','42','--simple-io','--grammar-file',str(gp),'-f',str(fp)],capture_output=True,timeout=240)
    raw=p.stdout.decode('utf8','replace').strip()
    if p.returncode:
        raise RuntimeError(p.stderr.decode('utf8','replace')[-1500:])
    log=p.stderr.decode('utf8','replace')
    if 'failed to parse grammar' in log.lower() or 'failed to parse grammar' in raw.lower():
        raise RuntimeError(log[-1500:])
    try:
        answer,cited = parse(raw,docs,q)
        error=''
    except (ValueError, KeyError, TypeError, ArithmeticError) as e:
        answer,cited,error='skip',[],str(e)
    return {"answer":answer,"cited":cited,"raw":raw,"error":error,"seconds":time.perf_counter()-started}


def fixed_subset(train, groups_path, per_fold=8):
    _, splits = base.group_folds(train, groups_path)
    groups = pd.read_csv(groups_path,dtype=str).set_index('qid').group
    ids = []
    for fold,(_,va) in enumerate(splits):
        candidates = sorted(train.iloc[va].qid, key=lambda q: hashlib.sha256(('pilot-cpu-2b-v1:'+q).encode()).digest())
        seen = set()
        for qid in candidates:
            if groups[qid] not in seen:
                ids.append(qid)
                seen.add(groups[qid])
                if len(seen) == per_fold:
                    break
    return ids


def evaluate(pred_path, train_path, baseline_path):
    scorer = base.module_from_file(base.find_file(base.ROOT,'score.py'))
    gold = {g['id']:g for g in scorer.load_gold([str(train_path)])}
    rows = [json.loads(s) for s in pred_path.read_text(encoding='utf8').splitlines()]
    baseline = pd.read_csv(baseline_path,dtype=str,keep_default_na=False).set_index('qid')
    sets = {key:[] for key in ('baseline','llm','hybrid','llm_fixed_baseline_evidence')}
    timings = []
    for row in rows:
        qid=row['qid'];g=gold[qid]
        fixed={'answer':baseline.at[qid,'answer'],'evidence':baseline.at[qid,'evidence'].split()}
        selected = row if row['answer'] not in ('skip',) and not row.get('error') else fixed
        for name, rec in zip(sets, [fixed,row,selected,dict(answer=row['answer'],evidence=fixed['evidence'] if row['answer']!='no_info' else [])]):
            sets[name].append(scorer.score_item(g,rec))
        timings.append(row['seconds'])
    import numpy as np
    report={k:scorer.summarize(v) for k,v in sets.items()}
    report['by_format']={k:{fmt:scorer.summarize([r for r in recs if gold[r['qid']]['answer_format']==fmt]) for fmt in sorted(set(gold[r['qid']]['answer_format'] for r in recs))} for k,recs in sets.items()}
    report.update(num_questions=len(rows),mean_seconds=float(np.mean(timings)),median_seconds=float(np.median(timings)),
                  p90_seconds=float(np.percentile(timings,90)),errors=sum(bool(r.get('error')) for r in rows))
    pred_path.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print([(k,report[k]['S']) for k in sets],f"mean wall time {np.mean(timings):.1f}s",flush=True)


def signature(exe, model, ranks, questions, corpus, args):
    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    behavior = [inspect.getsource(f) for f in (context, alternation, prompt_grammar, parse, infer,
                                                base.calculate, base.grounded_operand, base.normalize_answer)]
    behavior.extend((repr(MATH_OPS), repr(SPECIAL), repr(base.MATH_QUERY.pattern), DATE_LITERAL, NUMBER_LITERAL))
    return {"model": digest(model), "exe": digest(exe), "ranks": digest(ranks), "questions": digest(questions),
            "corpus": digest(corpus), "mode": args.mode, "threads": args.threads, "gpu_layers": args.gpu_layers,
            "behavior": hashlib.sha256("\n".join(behavior).encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('corpus','train','test','groups'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--output',type=Path,default=base.ROOT/'output_2b_final/cpu_llm')
    p.add_argument('--ranks',type=Path,default=base.ROOT/'output_2b_final/cpu_rank')
    p.add_argument('--model',type=Path,default=base.ROOT/'weights_2b/llama_cpp/qwen35-2b.gguf')
    p.add_argument('--exe',type=Path,default=base.ROOT/'weights_2b/llama_cpp/b12671/llama-completion.exe')
    p.add_argument('--adapt-ollama',type=Path)
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--gpu-layers',type=int,default=0)
    p.add_argument('--posts',type=int,default=6)
    p.add_argument('--excerpt-chars',type=int,default=600)
    p.add_argument('--ctx-size',type=int,default=3584)
    p.add_argument('--mode',choices=['pilot','train','test'],default='pilot')
    p.add_argument('--limit',type=int)
    p.add_argument('--format',choices=['number','date','status','yesno','choice'])
    p.add_argument('--baseline',type=Path,default=base.ROOT/'output_2b_final/cpu_lexical/oof.csv')
    args=p.parse_args()
    if args.adapt_ollama:
        adapt_ollama(args.adapt_ollama,args.model)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    part='test' if args.mode=='test' else 'train'
    path=base.find_file(base.ROOT,part+'.csv',getattr(args,part))
    questions=base.read_questions(path)
    if args.mode=='pilot':
        ids=fixed_subset(questions,base.find_file(base.ROOT,'train_groups.csv',args.groups))
        questions=questions.set_index('qid').loc[ids].reset_index()
    if args.format:
        questions=questions.loc[questions.answer_format==args.format]
    if args.limit:
        questions=questions.head(args.limit)
    target=args.output/(args.mode+'.jsonl')
    ranks_path=args.ranks/(part+'.jsonl')
    corpus_path=base.find_file(base.ROOT,'corpus.parquet',args.corpus)
    run_signature=signature(args.exe,args.model,ranks_path,path,corpus_path,args)
    run_signature.update(posts=args.posts,excerpt_chars=args.excerpt_chars,ctx_size=args.ctx_size)
    if args.format:
        run_signature['answer_format']=args.format
    fingerprint_path=target.with_suffix('.inputs.json')
    if fingerprint_path.exists():
        if json.loads(fingerprint_path.read_text()) != run_signature:
            raise ValueError('Reader checkpoint inputs differ. Select a new output directory.')
    elif target.exists():
        raise ValueError('Reader checkpoint lacks input fingerprint. Select a new output directory.')
    else:
        fingerprint_path.write_text(json.dumps(run_signature,indent=2))
    ranks={r['qid']:r for r in map(json.loads,ranks_path.read_text().splitlines())}
    posts=pd.read_parquet(corpus_path).set_index('post_id')
    old={r['qid']:r for r in map(json.loads,target.read_text(encoding='utf8').splitlines())} if target.is_file() else {}
    if old.keys()-set(questions.qid):
        raise ValueError('Existing predictions outside selected subset. Use a new output directory.')
    with tempfile.TemporaryDirectory() as td, target.open('a',encoding='utf8') as out:
        for i,q in enumerate(questions.to_dict('records')):
            qid=q['qid']
            if qid in old:
                continue
            rank=ranks[qid]
            docs=context(q,rank,posts,args.posts,args.excerpt_chars)
            pred=infer(args.exe.resolve(),args.model.resolve(),args.threads,args.gpu_layers,Path(td),q,docs,args.ctx_size)
            evidence=[] if pred['answer']=='no_info' else list(dict.fromkeys(pred['cited'] + rank['ids']))[:5]
            row={'qid':qid,'evidence':evidence,**pred}
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
            out.flush()
            print(f"{i+1}/{len(questions)} {qid} {pred['answer']} {pred.get('error','')} {pred['seconds']:.1f}s",flush=True)
    if part=='train' and args.baseline.is_file():
        evaluate(target,path,args.baseline)


if __name__=="__main__":
    main()


