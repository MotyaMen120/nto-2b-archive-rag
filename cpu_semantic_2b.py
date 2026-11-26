"""Offline ONNX semantic retrieval and cross-encoder features for the CPU evidence ranker."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import time
import tempfile

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from tokenizers import Tokenizer
from threadpoolctl import threadpool_limits

import cpu_rank_2b as lexical
import cpu_numeric_2b as numeric
import solution_2b_v2 as base

VERSION = "dense-ce-day-isolated-v2"
ROOT = base.ROOT
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def embedding_path(args, corpus_path, directory):
    signature = hashlib.sha256((VERSION+digest(corpus_path)+digest(directory/"model.onnx")+
                                digest(directory/"tokenizer.json")).encode()).hexdigest()[:20]
    return (args.cache_dir or args.output/"cache") / f"dense_{signature}.npy"


class Engine:
    def __init__(self, directory, threads, length):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(directory/"model.onnx"), so, providers=["CPUExecutionProvider"])
        self.keys = {x.name for x in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(directory/"tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=length)
        self.tokenizer.no_padding()
        self.output = self.session.get_outputs()[0].name
        self.pad = self.tokenizer.token_to_id("[PAD]")
        if self.pad is None:
            self.pad = self.tokenizer.token_to_id("<pad>")
        if self.pad is None:
            raise ValueError("Tokenizer has no padding token")

    def encode(self, inputs, batch_size, dense):
        if not inputs:
            return np.array([])
        encoded = self.tokenizer.encode_batch(inputs)
        order = sorted(range(len(inputs)), key=lambda i: len(encoded[i].ids))
        result = [None] * len(inputs)
        for offset in range(0, len(inputs), batch_size):
            chunks = order[offset:offset+batch_size]
            width = max(len(encoded[i].ids) for i in chunks)
            ids = np.full((len(chunks), width), self.pad, np.int64)
            mask = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for j, i in enumerate(chunks):
                item = encoded[i]
                length = len(item.ids)
                ids[j, :length] = item.ids
                mask[j, :length] = 1
                types[j, :length] = item.type_ids
            values = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            y = self.session.run([self.output], {k: values[k] for k in self.keys})[0]
            if dense:
                if y.ndim == 3:
                    if dense == "e5":
                        y = np.einsum("btd,bt->bd", y, mask.astype(np.float32)) / mask.sum(axis=1,keepdims=True)
                        y = y.astype(np.float32)
                    else:
                        y = y[:, 0]
                y = y / np.maximum(1.e-9, np.linalg.norm(y, axis=1, keepdims=True))
            else:
                y = y.reshape(-1)
            for j,i in enumerate(chunks):
                result[i] = y[j]
        return np.asarray(result)


def dense_search(args, corpus, frames, corpus_path, rank_dir, family):
    directory = args.dense_weights if family == "dense" else args.e5_weights
    cache = args.cache_dir or args.output/"cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = embedding_path(args,corpus_path,directory)
    engine = Engine(directory, args.threads, 128)
    docs = corpus.sort_values(["date","post_id"])
    days = np.array(docs.date, dtype="datetime64[D]")
    ids = docs.post_id.to_numpy(np.int64)
    channels = docs.channel.to_numpy()
    row_by_id = {p:i for i,p in enumerate(ids)}
    if not path.is_file():
        partial = path.with_suffix(".incomplete.npy")
        mapping = None
        # Quantized activation scales can be shared across a batch. Every passage
        # in one batch must therefore be public on the same day.
        stops = np.r_[np.flatnonzero(days[1:] != days[:-1]) + 1,len(days)]
        start = 0
        for stop in stops:
            texts = docs.text.iloc[start:stop].fillna("").tolist()
            if family == "e5":
                texts = ["passage: " + t for t in texts]
            vectors = engine.encode(texts, 32, family)
            if mapping is None:
                mapping = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                                    shape=(len(docs),vectors.shape[1]))
            mapping[start:start+len(texts)] = vectors
            if start // 8192 != stop // 8192 or start == 0:
                print(f"Dense embeddings {stop}/{len(docs)}", flush=True)
            start = stop
        mapping.flush()
        del mapping
        partial.replace(path)
    matrix = np.load(path,mmap_mode="r")
    for split, frame in frames.items():
        seeds = read_ranks(rank_dir/f"seed_{split}.jsonl")
        rows = frame.to_dict("records")
        prefix = "query: " if family == "e5" else ""
        questions = [prefix + base.rewrite(q)["canonical_query"] for q in rows]
        output = []
        for i,row in enumerate(rows):
            hi = int(np.searchsorted(days, np.datetime64(row["asked_at"]),side="right"))
            seed_idx = [row_by_id[p] for p in seeds[row["qid"]]["ids"]]
            if any(j >= hi for j in seed_idx):
                raise ValueError("Future document in feedback input")
            clauses = numeric._clauses(row["question"].lower().replace("ё","е"))
            use_hops = clauses[0] != clauses[1]
            query = engine.encode([questions[i]],1,family)[0]
            feedback = query.copy()
            if seed_idx:
                anchors = matrix[seed_idx[:4]].astype(np.float32)
                avg = np.average(anchors,axis=0,weights=[4,3,2,1][:len(anchors)])
                feedback = .45 * feedback + .55 * avg
                feedback /= max(1e-9,np.linalg.norm(feedback))
            vectors = [query,feedback]
            if use_hops:
                splits = engine.encode([prefix + c for c in clauses],2,family)
                vectors.extend(.65 * splits + .35 * query)
            vectors = np.stack(vectors).astype(np.float32)
            similarities = np.empty((hi,len(vectors)), np.float32)
            for start in range(0,hi,8192):
                stop = min(start+8192,hi)
                similarities[start:stop] = matrix[start:stop].astype(np.float32) @ vectors.T
            top = lexical.best(similarities[:,0], 80)
            feedback_group = lexical.best(similarities[:,1],45)
            hop_groups = [lexical.best(similarities[:,k],30) for k in range(2,len(vectors))]
            selected = np.unique(np.concatenate([top,feedback_group,np.asarray(seed_idx,dtype=np.int64),*hop_groups]))
            if np.any(selected>=hi):
                raise ValueError("Future evidence in lexical input")
            ordered = selected[np.lexsort((ids[selected],-similarities[selected,0]))]
            info = {"qid":row["qid"], "ids":ids[ordered].tolist(), "scores":similarities[ordered,0].tolist(),
                    "hop_top":[ids[idx[:6]].tolist() for idx in hop_groups],
                    "hop_scores":similarities[ordered,2:].tolist() if use_hops else [],
                    "feedback_top":ids[feedback_group[:8]].tolist(),"feedback_scores":similarities[ordered,1].tolist()}
            # A top candidate backed by other reports on the same day is often a better citation.
            if len(ordered):
                emb = matrix[ordered].astype(np.float32)
                sims = emb @ emb.T
                nearby = np.abs(days[ordered,None]-days[ordered][None,:]) <= np.timedelta64(2,"D")
                different_channel = channels[ordered,None] != channels[ordered][None,:]
                sims[~(nearby & different_channel)] = -1.
                info["cluster"] = np.column_stack([sims.max(axis=1),(sims>.75).sum(axis=1),
                                                  (sims>.88).sum(axis=1)]).tolist()
            output.append(info)
        lexical.dump_jsonl(args.output/f"{family}_{split}.jsonl",output)


def read_ranks(path):
    with path.open(encoding="utf8") as stream:
        return {r["qid"]: r for line in stream if (r := json.loads(line))}


def cross_encode(args, corpus, frames, corpus_path, rank_dir):
    ce = Engine(args.ce_weights, args.threads, 192)
    texts = corpus.set_index("post_id")["text"].to_dict()
    dates = {int(pid):str(day)[:10] for pid,day in zip(corpus.post_id,corpus.date)}
    model_key = VERSION+digest(corpus_path)+digest(args.ce_weights/"model.onnx")+digest(args.ce_weights/"tokenizer.json")
    for split, frame in frames.items():
        seed = read_ranks(rank_dir/f"seed_{split}.jsonl")
        families = ("dense","e5") if args.with_e5 else ("dense",)
        others = [read_ranks(path) for family in families
                  if (path := args.output/f"{family}_{split}.jsonl").exists()]
        records = []
        cache = (args.cache_dir or args.output/"cache")/"ce"
        cache.mkdir(parents=True,exist_ok=True)
        accumulated = 0
        start = time.perf_counter()
        for row in frame.to_dict("records"):
            qid = row["qid"]
            question = base.rewrite(row)["canonical_query"]
            ids = list(seed[qid]["ids"][:args.top])
            for ranks in others:
                ids.extend(ranks[qid]["ids"][:16])
                ids.extend(ranks[qid].get("feedback_top",[]))
                for anchors in ranks[qid].get("hop_top",[]):
                    ids.extend(anchors)
            ids = list(dict.fromkeys(ids))
            if any(dates[p] > row["asked_at"] for p in ids):
                raise ValueError("Future document at cross-encoder input")
            key = hashlib.sha256(json.dumps([model_key,question,row["asked_at"],ids],
                                           ensure_ascii=False).encode()).hexdigest()
            path = cache/f"{key}.json"
            if path.exists():
                scores = json.loads(path.read_text(encoding="utf8"))
            else:
                scores = ce.encode([(question,str(texts[p])) for p in ids],8,False).tolist()
                with tempfile.NamedTemporaryFile(dir=cache,delete=False,mode="w",encoding="utf8") as stream:
                    json.dump(scores,stream)
                    tmp = Path(stream.name)
                tmp.replace(path)
                accumulated += len(ids)
            records.append({"qid":qid,"ids":ids,"scores":scores})
            if len(records)%40==0:
                print(f"Cross-encoder {split} {len(records)}/{len(frame)}, {accumulated} new pairs in {time.perf_counter()-start:.1f}s",flush=True)
        lexical.dump_jsonl(args.output/f"ce_{split}.jsonl",records)


class CitationFeatures:
    def __init__(self,args,corpus,corpus_path):
        docs = corpus.sort_values(["date","post_id"])
        self.index = {p:i for i,p in enumerate(docs.post_id)}
        self.days = docs.date.to_numpy("datetime64[D]").astype(np.int32)
        self.channel = docs.channel.to_numpy()
        self.text = docs.text.fillna("").str.lower().str.replace("ё","е").to_numpy()
        self.vectors = np.load(embedding_path(args,corpus_path,args.dense_weights),mmap_mode="r")

    def features(self,question,ids,ce):
        result = np.zeros((len(ids),9),np.float32)
        docs = np.array([self.index[p] for p in ids],dtype=np.int64)
        if not len(docs):
            return result
        cutoff = np.datetime64(question["asked_at"],"D").astype(np.int32)
        if np.any(self.days[docs] > cutoff):
            raise ValueError("Future citation candidate")
        anchors = np.array([self.index[p] for p in sorted(ce,key=ce.get,reverse=True)[:15]],dtype=np.int64)
        if len(anchors):
            if np.any(self.days[anchors] > cutoff):
                raise ValueError("Future citation anchor")
            near = np.abs(self.days[docs,None]-self.days[anchors]) <= 2
            different = self.channel[docs,None] != self.channel[anchors]
            sim = self.vectors[docs].astype(np.float32) @ self.vectors[anchors].astype(np.float32).T
            strength = 1/np.sqrt(1+np.arange(len(anchors)))
            support = np.where(near & different,np.maximum(0,sim)*strength,0)
            result[:,0] = support.max(axis=1)
            result[:,1] = support.sum(axis=1)
            result[:,2] = np.maximum(0,np.where(docs[:,None]!=anchors,sim*strength,0)).max(axis=1)
            result[:,3] = (near & different).dot(strength)
            result[:,4] = (near & different & (sim>.66)).sum(axis=1)
        unit = question.get("unit")
        pattern = numeric._unit_spec(unit)[1] if unit else None
        numerals = set(re.findall(r"\d+(?:[.,]\d+)?",question["question"]))
        for i,d in enumerate(docs):
            text = self.text[d]
            found = re.findall(r"\d+(?:[.,]\d+)?",text)
            result[i,5] = np.log1p(len(found))
            result[i,6] = len(numerals.intersection(found))
            result[i,7] = min(8,len(pattern.findall(text))) if pattern else 0
            result[i,8] = bool(re.search(r"\b(?:планир\w*|обеща\w*|собира\w*|запланир\w*)",text))
        return result


def inputs(split, frame, args, corpus, citation=None):
    raw = np.load(args.rank_dir/f"features_{split}.npz",allow_pickle=False)
    if raw["qids"].tolist() != frame.qid.tolist():
        raise ValueError("Ranked questions do not match")
    cuts = np.cumsum(raw["sizes"])[:-1]
    raw_ids, raw_x = np.split(raw["ids"],cuts), np.split(raw["features"],cuts)
    ce = read_ranks(args.output/f"ce_{split}.jsonl")
    families = ("dense","e5") if args.with_e5 else ("dense",)
    dense_ranks = [read_ranks(path) for family in families
                   if (path := args.output/f"{family}_{split}.jsonl").exists()]
    all_ids, all_x = [], []
    for i, row in enumerate(frame.to_dict("records")):
        qid = row["qid"]
        c = ce[qid]
        cs = dict(zip(c["ids"],c["scores"]))
        sources = [rank[qid] for rank in dense_ranks]
        ids = np.asarray(list(dict.fromkeys(raw_ids[i].tolist()+[p for d in sources for p in d["ids"]])),np.int64)
        x = np.full((len(ids),raw_x[i].shape[1]),np.nan,np.float32)
        x[:len(raw_ids[i])] = raw_x[i]
        extra = np.full((len(ids),4+9*len(sources)),np.nan,np.float32)
        for j,p in enumerate(ids):
            age = (row["_day"]-corpus[p]).days
            if age<0:
                raise ValueError("Future semantic document")
            extra[j,0] = np.log1p(age)
        lookups = [(1,cs)] + [(4+9*k,dict(zip(d["ids"],d["scores"]))) for k,d in enumerate(sources)]
        for off, score_map in lookups:
            order = sorted(score_map,key=score_map.get,reverse=True)
            places = {p:j+1 for j,p in enumerate(order)}
            for j,p in enumerate(ids):
                if p in score_map:
                    v = score_map[p]
                    extra[j,off:off+3] = [v, v-score_map[order[0]], 1/places[p]]
        positions = {p:j for j,p in enumerate(ids)}
        for k,d in enumerate(sources):
            off = 7+9*k
            rows = [positions[p] for p in d["ids"]]
            if d.get("hop_scores"):
                extra[rows,off:off+2] = d["hop_scores"]
            if d.get("cluster"):
                extra[rows,off+2:off+5] = d["cluster"]
            if d.get("feedback_scores"):
                extra[rows,off+5] = d["feedback_scores"]
        all_ids.append(ids)
        blocks = [x,extra]
        if citation is not None:
            blocks.append(citation.features(row,ids,cs))
        all_x.append(np.column_stack(blocks))
    return all_ids, all_x


def fuse(args, corpus, frames, train_path, groups_path):
    from datetime import date
    days = {p:date.fromisoformat(str(d)[:10]) for p,d in zip(corpus.post_id,corpus.date)}
    corpus_path = base.find_file(ROOT,"corpus.parquet",args.corpus)
    model = joblib.load(args.model) if args.model else None
    n_basic = lexical.FEATURE_COUNT + 4 + 9*(1+int(args.with_e5))
    use_citations = (model.n_features_in_ > n_basic) if model is not None else not args.base_fusion
    citation = CitationFeatures(args,corpus,corpus_path) if use_citations else None
    data = {}
    for split,frame in frames.items():
        frame = frame.assign(_day=frame.asked_at.apply(date.fromisoformat))
        data[split] = inputs(split,frame,args,days,citation)
    if model is None:
        train = frames["train"]
        ids, features = data["train"]
        scorer = base.module_from_file(base.find_file(ROOT,"score.py"))
        gold = {r["id"]:r for r in scorer.load_gold([str(train_path)])}
        _,splits = base.group_folds(train,groups_path)
        oof = [None]*len(train)
        for tr,va in splits:
            model = lexical.ranker(features,ids,train,gold,tr,args.threads)
            for i,r in zip(va,lexical.apply(model,train,ids,features,va,args.threads)):
                oof[i]=r
        lexical.dump_jsonl(args.output/"train.jsonl",oof)
        diagnostics = {}
        for lev in ("all","light","medium","heavy"):
            sample = [(gold[r["qid"]],r["ids"]) for r in oof if gold[r["qid"]]["_G"]
                      and (lev == "all" or gold[r["qid"]]["level"] == lev)]
            w = [g["_w"] for g,_ in sample]
            ap = [scorer.ap5(ids[:5],g["_G"]) for g,ids in sample]
            diagnostics[lev] = {"weighted_ap5":float(np.average(ap,weights=w))}
            for k in (5,20,100):
                recall = [len(set(ids[:k]) & set(g["_G"]))/len(set(g["_G"])) for g,ids in sample]
                diagnostics[lev][f"hit{k}"] = float(np.mean(np.array(recall)>0))
                diagnostics[lev][f"recall{k}"] = float(np.mean(recall))
        (args.output/"evidence.report.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf8")
        print("Grouped evidence metrics",diagnostics["all"],flush=True)
        model = lexical.ranker(features,ids,train,gold,range(len(train)),args.threads)
        joblib.dump(model,args.output/"model.joblib")
    ids, features = data["test"]
    lexical.dump_jsonl(args.output/"test.jsonl",lexical.apply(model,frames["test"],ids,features,
                        range(len(frames["test"])),args.threads))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["dense","e5","ce","fuse","all"])
    for name in ["corpus","train","test","groups","model"]:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"output_2b_final/cpu_semantic")
    parser.add_argument("--rank-dir",type=Path,default=ROOT/"output_2b_final/cpu_rank")
    parser.add_argument("--cache-dir",type=Path,help="Shared content-addressed embedding and pair cache")
    parser.add_argument("--ce-weights",type=Path,default=ROOT/"weights_2b/cpu_ce")
    parser.add_argument("--dense-weights",type=Path,default=ROOT/"weights_2b/cpu_dense")
    parser.add_argument("--e5-weights",type=Path,default=ROOT/"weights_2b/cpu_e5")
    parser.add_argument("--with-e5", action="store_true",help="Use the second dense encoder in a full run")
    parser.add_argument("--base-fusion", action="store_true",help="Train without citation features for comparison")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--top",type=int,default=32)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    corpus_path=base.find_file(ROOT,"corpus.parquet",args.corpus)
    corpus=pd.read_parquet(corpus_path,columns=["post_id","date","channel","text"])
    frames={"test":base.read_questions(base.find_file(ROOT,"test.csv",args.test))}
    train_path=groups_path=None
    if not args.model:
        train_path=base.find_file(ROOT,"train.csv",args.train)
        frames["train"]=base.read_questions(train_path)
        groups_path=base.find_file(ROOT,"train_groups.csv",args.groups)
    with threadpool_limits(limits=1,user_api="blas"):
        if args.stage in ("dense","all"):
            dense_search(args,corpus,frames,corpus_path,args.rank_dir,"dense")
        if args.stage == "e5" or (args.stage == "all" and args.with_e5):
            dense_search(args,corpus,frames,corpus_path,args.rank_dir,"e5")
        if args.stage in ("ce","all"):
            cross_encode(args,corpus,frames,corpus_path,args.rank_dir)
        if args.stage in ("fuse","all"):
            fuse(args,corpus,frames,train_path,groups_path)


if __name__=="__main__":
    main()
