"""Train a Russian BERT to verify numbers, dates, and two timeline endpoints in archive posts."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from safetensors.torch import load_file

import cpu_numeric_2b as num
import cpu_timeline_2b as timeline
import solution_2b_v2 as base
from predict_offline_2b import check_dates, digest
from run_cpu_2b import load_corpus, write_checked


@dataclass
class Candidate:
    role: int
    value: str
    pid: int
    rank: int
    hint: float
    explicit: bool
    tokens: tuple


class Verifier(nn.Module):
    """Load the three published BERT layers into PyTorch's standard encoder layers."""
    def __init__(self, path):
        super().__init__()
        config = json.loads((path / "config.json").read_text(encoding="utf8"))
        weights = load_file(str(path / "model.safetensors"))
        hidden = config["hidden_size"]
        eps = config["layer_norm_eps"]
        embeddings = "bert.embeddings."
        self.words, self.positions, self.types = (
            nn.Embedding.from_pretrained(weights[embeddings + name + ".weight"], freeze=True)
            for name in ("word_embeddings", "position_embeddings", "token_type_embeddings"))
        self.norm = nn.LayerNorm(hidden, eps=eps)
        self.norm.load_state_dict({k: weights[embeddings + "LayerNorm." + k] for k in ("weight", "bias")})
        self.layers = nn.ModuleList()
        for index in range(config["num_hidden_layers"]):
            layer = nn.TransformerEncoderLayer(
                hidden, config["num_attention_heads"], config["intermediate_size"],
                dropout=0.0, activation="gelu", layer_norm_eps=eps, batch_first=True)
            prefix = f"bert.encoder.layer.{index}."
            state = {}
            for key in ("weight", "bias"):
                state["self_attn.in_proj_" + key] = torch.cat(
                    [weights[prefix + "attention.self." + component + "." + key]
                     for component in ("query", "key", "value")])
                for src, dst in (("attention.output.dense", "self_attn.out_proj"),
                                 ("attention.output.LayerNorm", "norm1"),
                                 ("intermediate.dense", "linear1"), ("output.dense", "linear2"),
                                 ("output.LayerNorm", "norm2")):
                    state[dst + "." + key] = weights[prefix + src + "." + key]
            layer.load_state_dict(state)
            self.layers.append(layer)
        self.pool = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 2)
        for layer, name in ((self.pool, "bert.pooler.dense"), (self.head, "classifier")):
            layer.load_state_dict({key: weights[name + "." + key] for key in ("weight", "bias")})

    def logits(self, ids, types):
        pad = ids.eq(0)
        x = self.norm(self.words(ids) + self.types(types)
                      + self.positions(torch.arange(ids.shape[1], device=ids.device)))
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad)
        return self.head(torch.tanh(self.pool(x[:, 0])))

    def forward(self, ids, types):
        logits = self.logits(ids, types)
        return logits[:, 1] - logits[:, 0]


def batch(items):
    width = max(len(c.tokens[0]) for c in items)
    ids = torch.zeros((len(items), width), dtype=torch.int64)
    types = torch.zeros_like(ids)
    for i, candidate in enumerate(items):
        tokens, segments = candidate.tokens
        ids[i, :len(tokens)] = torch.as_tensor(tokens, dtype=torch.int64)
        types[i, :len(tokens)] = torch.as_tensor(segments, dtype=torch.int64)
    return ids, types


def endpoint_gap(a, b, row):
    if not a < b:
        return None
    if "месяц" in (row.get("unit") or ""):
        whole = "полн" in (row["question"] + row["unit"]).lower()
        return str((b.year-a.year)*12 + b.month-a.month - int(whole and b.day < a.day))
    return str((b-a).days)


def mentions(row, ids, corpus, tokenizer):
    docs = []
    for pid in ids:
        post = corpus[pid]
        if post["date"] > row["asked_at"]:
            raise ValueError("Future post in mention reader")
        docs.append(post)
    parsed = num._documents(row, docs, limit=len(docs))
    question = num._norm(row["question"])
    unit = row.get("unit") or ""
    interval = timeline.supported(row)
    events = timeline.clauses(question) if interval else (question,)
    results = []
    words = num._words(question)
    for d, post in zip(parsed, docs):
        for role, target in enumerate(events):
            if row["answer_format"] == "number" and not interval:
                measurements = sorted(num._mentions(d, words, words, num._norm(unit)),
                                      key=lambda m: -m.strength)[:8]
                items = [(num._number(m.value), m.start, m.end, m.hint, True) for m in measurements]
                subject = row["question"] + " В каких единицах: " + unit
            else:
                subject = "Какова дата события «" + target + "»?" if interval else row["question"]
                moments = [*d.moments, *timeline._extra_dates(d.text, d.pub)]
                items = [(d.pub.isoformat(), 0, 0, 0.0, False)]
                items.extend((m.value.isoformat(), m.start, m.end, 1.0, True) for m in moments[:12])
                if not interval and num.FIRST_NEWS.search(question):
                    items = items[:1]
            for value, start, end, hint, explicit in items:
                if explicit:
                    context = (d.text[max(0, start-150):start] + " [ " + d.text[start:end] +
                               " ] " + d.text[end:end+145])
                else:
                    context = "Дата публикации " + post["date"] + ". " + d.text[:275]
                claim = subject + " Предложенный ответ: " + value
                pair = tokenizer.encode(claim, context)
                results.append(Candidate(role, value, post["post_id"], d.rank, hint, explicit,
                                         (np.asarray(pair.ids, dtype=np.int32),
                                          np.asarray(pair.type_ids, dtype=np.int8))))
    return results


def weak_targets(row, candidates):
    evidence = {int(p) for p in row["evidence"].split()}
    if timeline.supported(row):
        possible = [set(), set()]
        for c in candidates:
            if c.pid in evidence:
                possible[c.role].add(date.fromisoformat(c.value))
        matches = set()
        for start in possible[0]:
            for end in possible[1]:
                gap = endpoint_gap(start, end, row)
                if gap is not None and num._match(gap, row["answer"], "number"):
                    matches.update(((0, start.isoformat()), (1, end.isoformat())))
        return [c.pid in evidence and (c.role, c.value) in matches for c in candidates]
    return [c.pid in evidence and num._match(c.value, row["answer"], row["answer_format"])
            for c in candidates]


def fit(rows, contexts, weights, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    model = Verifier(weights)
    model.layers[0].requires_grad_(False)
    model.norm.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": model.layers[1:].parameters(), "lr": 4e-5},
        {"params": [*model.pool.parameters(), *model.head.parameters()], "lr": 9e-5},
    ], weight_decay=.01)
    examples = []
    for row in rows:
        candidates = contexts.get(row["qid"], [])
        labels = weak_targets(row, candidates)
        for role in {c.role for c in candidates}:
            positives = [c for c, y in zip(candidates, labels) if c.role == role and y]
            negatives = [c for c, y in zip(candidates, labels) if c.role == role and not y]
            if positives and negatives:
                examples.append((positives, negatives))
    for _ in range(3):
        training = []
        for i in rng.permutation(len(examples)):
            positives, negatives = examples[i]
            # Give each question and endpoint the same total weight.
            a = rng.integers(len(positives), size=min(3, len(positives)))
            b = rng.choice(len(negatives), size=min(4, len(negatives)), replace=False)
            training.extend((positives[j], 1.0, 1/len(a)) for j in a)
            training.extend((negatives[j], 0.0, 1/len(b)) for j in b)
        rng.shuffle(training)
        for offset in range(0, len(training), 24):
            chunk = training[offset:offset+24]
            cs, labels, scale = zip(*chunk)
            prediction = model(*batch(cs))
            loss = nn.functional.binary_cross_entropy_with_logits(
                prediction, torch.tensor(labels), weight=torch.tensor(scale, dtype=torch.float32))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def answer(row, candidates, model):
    if not candidates:
        return None
    logits = []
    for pos in range(0, len(candidates), 32):
        logits.extend(model(*batch(candidates[pos:pos+32])).tolist())
    roles = [defaultdict(dict), defaultdict(dict)]
    for c, logit in zip(candidates, logits):
        strength = float(logit) - .34*np.log1p(c.rank) + .17*c.hint
        posts = roles[c.role][c.value]
        posts[c.pid] = max(posts.get(c.pid, -np.inf), strength)
    def ranked(table):
        vals = []
        for value, posts in table.items():
            top = sorted(posts.values(), reverse=True)[:3]
            pooled = top[0] + .26*np.log(sum(np.exp(x-top[0]) for x in top))
            vals.append((pooled, value))
        return sorted(vals, reverse=True)
    left = ranked(roles[0])
    if not timeline.supported(row):
        return left[0][1]
    pairs = []
    for score_a, a in left[:18]:
        for score_b, b in ranked(roles[1])[:18]:
            gap = endpoint_gap(date.fromisoformat(a), date.fromisoformat(b), row)
            if gap is not None:
                pairs.append((score_a + score_b, gap))
    return max(pairs)[1] if pairs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "train", "test", "groups", "model"):
        parser.add_argument("--" + name, type=Path)
    root = base.ROOT / "output_2b_final"
    parser.add_argument("--rank-dir", type=Path, default=root/"cpu_large_ce")
    parser.add_argument("--seed-dir", type=Path, default=root/"cpu_rank")
    parser.add_argument("--base-dir", type=Path, default=root/"cpu_large_reader")
    parser.add_argument("--weights", type=Path, default=base.ROOT/"weights_2b/nli_tiny")
    parser.add_argument("--output", type=Path, default=root/"cpu_mention/submission.csv")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = args.output.parent
    out.mkdir(exist_ok=True, parents=True)
    tokenizer = Tokenizer.from_file(str(args.weights/"tokenizer.json"))
    tokenizer.enable_truncation(max_length=112)
    corpus_path = base.find_file(base.ROOT, "corpus.parquet", args.corpus)
    corpus = load_corpus(corpus_path)

    def read(split):
        path = base.find_file(base.ROOT, split + ".csv", getattr(args, split))
        frame = base.read_questions(path)
        ranks = {r["qid"]: r["ids"] for r in map(json.loads,
                 (args.rank_dir/(split+".jsonl")).read_text(encoding="utf8").splitlines())}
        records = frame.to_dict("records")
        relevant = [row for row in records if row["answer_format"] in ("date", "number")]
        inputs = {row["qid"]: mentions(row, ranks[row["qid"]][:9], corpus, tokenizer) for row in relevant}
        return path, frame, records, relevant, inputs

    def predict(model, rows, inputs, baselines):
        result = []
        for row in rows:
            old = baselines[row["qid"]]
            if row["qid"] in inputs and old["answer"] not in ("no_info", "promised"):
                value = answer(row, inputs[row["qid"]], model)
                if value is not None:
                    old = dict(old, answer=value)
            result.append(old)
        return result

    from cpu_span_2b import _records
    if not args.model:
        train_path, frame, rows, relevant, valid_inputs = read("train")
        groups_path = base.find_file(base.ROOT, "train_groups.csv", args.groups)
        folds, splits = base.group_folds(frame, groups_path)
        seed = {r["qid"]: r["ids"] for r in map(json.loads,
                (args.seed_dir/"seed_train.jsonl").read_text(encoding="utf8").splitlines())}
        training = {}
        for row in relevant:
            gold = [int(p) for p in row["evidence"].split() if corpus[int(p)]["date"] <= row["asked_at"]]
            ids = list(dict.fromkeys([*gold[:8], *seed[row["qid"]][:6]]))
            training[row["qid"]] = mentions(row, ids, corpus, tokenizer)
        baselines = {r["qid"]: r for r in _records(args.base_dir/"oof.csv")}
        preds = {}
        for fold, (tr, va) in enumerate(splits, 1):
            model = fit([rows[i] for i in tr], training, args.weights, args.threads)
            for result in predict(model, [rows[i] for i in va], valid_inputs, baselines):
                preds[result["qid"]] = result
            print(f"Mention reader fold {fold}/5 complete", flush=True)
        oof = out / "oof.csv"
        write_checked(oof, [preds[row["qid"]] for row in rows], train_path, corpus_path)
        check_dates(oof, train_path, corpus_path)
        scorer = base.module_from_file(base.find_file(base.ROOT, "score.py"))
        local, report = scorer.run(str(oof), [str(train_path)])
        report["fold_S"] = [scorer.total([r for i,r in enumerate(local) if folds[i] == k]) for k in range(5)]
        (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(f"Official grouped S = {report['S']:.4f}", flush=True)
        model = fit(rows, training, args.weights, args.threads)
        torch.save({"state": model.state_dict(), "tokenizer": digest(args.weights/"tokenizer.json")}, out/"mention.pt")
        del training, valid_inputs
    else:
        snapshot = torch.load(args.model, weights_only=True, map_location="cpu")
        if snapshot["tokenizer"] != digest(args.weights/"tokenizer.json"):
            raise ValueError("Tokenizer differs from the model training configuration")
        model = Verifier(args.weights)
        model.load_state_dict(snapshot["state"])
        model.eval()
    test_path, _, rows, _, test_inputs = read("test")
    baselines = {r["qid"]: r for r in _records(args.base_dir/"submission.csv")}
    write_checked(args.output, predict(model, rows, test_inputs, baselines), test_path, corpus_path)
    check_dates(args.output, test_path, corpus_path)


if __name__ == "__main__":
    main()
