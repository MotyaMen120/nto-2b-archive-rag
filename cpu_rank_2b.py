"""CPU evidence retrieval and grouped LambdaRank for the offline archive task."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from datetime import date
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import solution_2b_v2 as base


WORDS = re.compile(r"[а-яёa-z]+|\d+(?:[.,]\d+)?", re.I)
STOP = base.STOP | set("""через что чтобы только также сейчас свой свою своих той там тут кто уже еще ещё какой каком какую какие
скольких насколько которую который когда сколько каким каких которых котором какая какое такие такой своей
этот этом этом этого этих этом той том того той году год годах года лет ли по это после про нельзя можно было была были был будет
вообще новый новости новость впервые сообщали сообщили последним данным правда""".lower().split())
MONTH = "январ феврал март апрел ма[йея] июн июл август сентябр октябр ноябр декабр".split()
MONTH_PATTERN = "|".join(f"({prefix}[а-я]*)" for prefix in MONTH)
MONTH_DATE = re.compile(r"(?:(\d{1,2})\s+)?(?:" + MONTH_PATTERN + r")(?:\s+(20\d\d))?\b")
EXTRA = [
    (r"город[а-я]*\s+(?:атомщик|ядерщик)|закрыт[а-я]*\s+город", "Саров Саровский"),
    (r"город[а-я]*\s+ножовщик", "Павлово Павловский"),
    (r"город[а-я]*\s+металлург", "Выкса Выксунский"),
    (r"город[а-я]*\s+корабел|город[а-я]*\s+судостроител", "Навашино Городец"),
    (r"город[а-я]*\s+стекло", "Бор Борский"),
    (r"город[а-я]*\s+оружейник", "Павлово"),
    (r"подземк|подземн[а-я]+ транспорт", "метро метрополитен"),
    (r"рельс[а-я]*\s+(?:к\s+)?крепост|подъ[её]м[а-я]*\s+по\s+рельсам", "фуникулер кремль"),
    (r"главн[а-я]+ пешеходн", "Большая Покровская Покровка"),
    (r"площад[а-я]*\s+дв[уа]х?\s+революц", "площадь Минина Пожарского"),
    (r"Нижн[яе]м?\b", "Нижний Новгород"),
    (r"\b(?:руб|оплат|стоит|цен[а-я]+|расценк|ценник)", "стоимость рублей цена"),
    (r"электросамокат", "самокат"),
    (r"ледов[а-я]+\s+арен|ледов[а-я]+\s+двор", "хоккей"),
]
FORMATS = ("number", "date", "choice", "yesno", "status")


def targets(question, asked_at):
    result = set(base.extract_dates(question))
    cutoff = date.fromisoformat(asked_at)
    for m in MONTH_DATE.finditer(question.lower()):
        number, *rest = m.groups()
        month = 1 + next(i for i, value in enumerate(rest[:12]) if value)
        year = int(rest[12]) if rest[12] else cutoff.year - int(month > cutoff.month)
        try:
            result.add(date(year, month, int(number or 15)).isoformat())
        except ValueError:
            pass
    return np.array(sorted(result), dtype="datetime64[D]").astype(np.int32)


class Archive:
    def __init__(self, posts):
        import pymorphy3
        from nltk.stem.snowball import RussianStemmer
        self.morph = pymorphy3.MorphAnalyzer()
        self.stem = RussianStemmer().stem
        data = posts.sort_values(["date", "post_id"], kind="stable").reset_index(drop=True)
        self.ids = data.post_id.to_numpy(np.int64)
        self.days = data.date.to_numpy(dtype="datetime64[D]").astype(np.int32)
        self.channels = pd.factorize(data.channel)[0]
        self.kinds = data.channel_type.astype(str).to_numpy()
        self.texts = data.text.astype(str).to_numpy()
        self.vocab = {}
        indptr, columns, values = [0], array("i"), array("f")
        for text, channel in zip(self.texts, data.channel):
            terms = self.words(str(channel) + " " + text)
            counts = Counter(self.vocab.setdefault(word, len(self.vocab)) for word in terms)
            columns.extend(counts)
            values.extend(counts.values())
            indptr.append(len(columns))
        self.docs = csr_matrix((values, columns, indptr), shape=(len(self.ids), len(self.vocab)), dtype=np.float32)
        self.docs.sort_indices()
        self.index = self.docs.tocsc()
        self.length = np.asarray(self.docs.sum(axis=1)).ravel()
        self.total_length = self.length.cumsum(dtype=np.float64)
        self.doc_terms = np.diff(self.docs.indptr).clip(1)
        print(f"Archive: {len(data):,} posts and {len(self.vocab):,} terms", flush=True)

    @lru_cache(maxsize=400000)
    def normalize(self, word):
        if word.isdecimal() or not ('а' <= word[0] <= 'я'):
            return word.replace(",", ".")
        lemma = self.morph.parse(word)[0].normal_form.replace("ё", "е")
        return self.stem(lemma)

    def words(self, text):
        words = WORDS.findall(text.lower().replace("ё", "е"))
        return [self.normalize(w) for w in words if w not in STOP and len(w) > int(not w.isdecimal())]

    def terms(self, text, hi):
        result = set()
        for word in self.words(text):
            term = self.vocab.get(word)
            if term is not None and self.index.indices[self.index.indptr[term]] < hi:
                result.add(term)
        return result

    def postings(self, term, hi):
        start, stop = self.index.indptr[term:term + 2]
        idx = self.index.indices[start:stop]
        count = idx.searchsorted(hi)
        return idx[:count], self.index.data[start:start + count]

    def score(self, terms, hi, norm, cache):
        scores = np.zeros(hi, np.float32)
        total = 0
        for term, weight in terms.items():
            if term not in cache:
                idx, tf = self.postings(term, hi)
                idf = math.log1p((hi - len(idx) + 0.5) / (len(idx) + 0.5))
                cache[term] = (idx, idf * (2.2 * tf) / (tf + norm[idx]), idf)
            idx, contribution, idf = cache[term]
            scores[idx] += weight * contribution
            total += weight * idf
        return scores, max(1, total)

    def candidates(self, q):
        day = np.datetime64(q["asked_at"], "D").astype(np.int32)
        hi = int(self.days.searchsorted(day, side="right"))
        if not hi:
            return np.array([], np.int32), np.empty((0, FEATURE_COUNT), np.float32), np.empty(0)
        days = self.days[:hi]
        age = day - days
        dates = targets(q["question"], q["asked_at"])
        distance = np.minimum.reduce([abs(days - target) for target in dates]) if len(dates) else np.full(hi, 10000)
        norm = 1.2 * (0.55 + 0.45 * self.length[:hi] / max(1, self.total_length[hi - 1] / hi))
        cache = {}
        def score(terms):
            return self.score(dict.fromkeys(terms, 1.0), hi, norm, cache)

        question = q["question"]
        plan = base.rewrite(q)
        alias = " ".join([plan["canonical_query"][len(question):]] + [e for pat, e in EXTRA if re.search(pat, question, re.I)])
        original = self.terms(question, hi)
        extra = self.terms(alias, hi) - original
        raw, raw_idf = score(original)
        aliased, _ = score(extra)
        options = self.terms(q["options"], hi) - original - extra if q.get("options") else set()
        option, _ = score(options)
        lexical = raw + 0.78 * aliased + 0.42 * option
        near = np.exp(-age / 22)
        near_target = np.exp(-distance / 24)
        initial = lexical * (1 + 0.60 * near + 0.45 * near_target)
        top = best(initial, 10)
        feedback = Counter()
        for position, doc in enumerate(top):
            ts = self.docs.indices[self.docs.indptr[doc]:self.docs.indptr[doc + 1]]
            for term in ts:
                if term not in original and term not in extra and term not in options:
                    feedback[int(term)] += 1 / (1 + position * 0.15)
        def importance(pair):
            term, freq = pair
            posts, _ = self.postings(term, hi)
            return (freq * math.log((hi+1)/(len(posts)+1)), -term)
        expansion = [t for t, _ in sorted(feedback.items(), key=importance, reverse=True)[:10]]
        related, _ = score(expansion)
        adjusted = (lexical + related * 0.38) * (1 + 0.55 * near + 0.40 * near_target)

        segment = np.zeros(hi, np.float32)
        parts = re.split(r"(?:\bмежду\b|\bдо того как\b|\bи (?:самим|фактическ|новостью|сообщением|реальным)|\s+или\s+|\s+и\s+(?=\S+\s+(?:на|в|у))|[;?])", question, flags=re.I)
        if len(parts) > 1:
            for part in parts:
                terms = self.terms(part, hi)
                if len(terms) >= 3 and terms != original:
                    values, idf = score(terms | extra)
                    np.maximum(segment, values / math.sqrt(idf), out=segment)
        candidates = np.unique(np.concatenate([best(lexical, 100), best(initial, 140), best(adjusted, 160),
                                               best(segment * (1 + near * .45), 70) if segment.any() else [],
                                               best(raw * (1 + 2 * near + near_target), 100)]).astype(np.int32))
        d = candidates
        local_docs = self.docs[d].copy()
        # IDF for the *eligible* postings only. Repost similarity is restricted to visible candidates.
        unique_terms, reversed_terms = np.unique(local_docs.indices, return_inverse=True)
        df = np.array([len(self.postings(t, hi)[0]) for t in unique_terms], dtype=np.float32)
        local_docs.data = (np.log1p(local_docs.data) * np.log1p(hi / (1 + df))[reversed_terms]).astype(np.float32)
        sim = (local_docs @ local_docs.T).toarray()
        scale = np.sqrt(sim.diagonal()).clip(.001)
        sim /= scale[:, None] * scale[None, :]
        different = self.channels[d, None] != self.channels[d][None, :]
        window = (np.abs(days[d, None] - days[d][None, :]) <= 2) & different
        np.fill_diagonal(sim, 0)
        local_sim = sim * window
        rel = initial[d] / max(1, initial.max())
        support = local_sim * rel[None, :]
        topic = sim * rel[None, :]
        seed = (adjusted[d] / max(1, adjusted.max()) * .6 + rel * .4 +
                .12 * support.max(axis=1) + .04 * np.minimum((local_sim > .22).sum(axis=1), 4))
        def columns(values):
            return [values[d], values[d] / max(1, values.max())]
        features = [*columns(raw), raw[d] / raw_idf, *columns(aliased), *columns(option), *columns(lexical),
                    *columns(initial), *columns(related), *columns(adjusted), *columns(segment), seed,
                    np.log1p(age[d]), np.log1p(distance[d]), self.length[d], self.doc_terms[d],
                    scale, (local_sim > .15).sum(axis=1), (local_sim > .30).sum(axis=1), (local_sim > .55).sum(axis=1),
                    support.max(axis=1), support.sum(axis=1), topic.max(axis=1), topic.sum(axis=1),
                    *[(self.kinds[d] == typ) for typ in ("Паблик", "СМИ", "Госструктура")],
                    *[np.full(len(d), q["answer_format"] == fmt) for fmt in FORMATS],
                    np.full(len(d), len(original)), np.full(len(d), len(extra)), np.full(len(d), len(options))]
        return d, np.column_stack(features).astype(np.float32), seed


FEATURE_COUNT = 41


def best(scores, k):
    if len(scores) <= k:
        idx = np.arange(len(scores))
    else:
        idx = np.argpartition(scores, -k)[-k:]
    return idx[np.lexsort((idx, -scores[idx]))]


def prediction(qid, ids, scores):
    order = np.lexsort((ids, -scores))
    return {"qid": qid, "ids": ids[order].astype(int).tolist(), "scores": scores[order].astype(float).round(5).tolist()}


def dump_jsonl(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def collect(archive, question_frame, output, prefix):
    matrices, candidates, seed = [], [], []
    for i, q in enumerate(question_frame.to_dict("records")):
        indices, features, scores = archive.candidates(q)
        if features.shape[1] != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} features, found {features.shape[1]}.")
        ids = archive.ids[indices]
        candidates.append(ids)
        matrices.append(features)
        seed.append(prediction(q["qid"], ids, scores))
        if (i + 1) % 100 == 0:
            print(f"{prefix} {i + 1}/{len(question_frame)}", flush=True)
    dump_jsonl(output / f"seed_{prefix}.jsonl", seed)
    np.savez_compressed(output / f"features_{prefix}.npz", qids=question_frame.qid.to_numpy(str),
                        sizes=np.array([len(ids) for ids in candidates]), ids=np.concatenate(candidates),
                        features=np.vstack(matrices))
    return candidates, matrices


def ranker(matrices, ids, train_rows, gold, indices, threads):
    from lightgbm import LGBMRanker
    x, y, sizes, weights = [], [], [], []
    for i in indices:
        row = gold[train_rows.iloc[i].qid]
        if not len(ids[i]):
            continue
        label = np.isin(ids[i], row["_G"]).astype(np.int8)
        if label.any():
            x.append(matrices[i])
            y.append(label)
            sizes.append(len(label))
            weights.extend([row["_w"]] * len(label))
    return LGBMRanker(n_estimators=180, num_leaves=15, learning_rate=.045, min_child_samples=45,
                      reg_lambda=6, metric="map", random_state=42, n_jobs=threads,
                      verbosity=-1, deterministic=True, force_col_wise=True).fit(
                          np.vstack(x), np.concatenate(y), group=sizes, sample_weight=np.asarray(weights))


def apply(model, questions, ids, matrices, indices, threads):
    result = []
    for i in indices:
        scores = model.booster_.predict(matrices[i], num_threads=threads) if len(ids[i]) else np.zeros(0)
        result.append(prediction(questions.iloc[i].qid, ids[i], scores))
    return result


def main():
    import joblib
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("corpus", "train", "test", "groups"):
        p.add_argument("--" + arg, type=Path)
    p.add_argument("--output", type=Path, default=base.ROOT / "output_2b_final/cpu_rank")
    p.add_argument("--model", type=Path, help="Use a fitted ranker; produce test predictions only")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    begin = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    test = base.read_questions(base.find_file(base.ROOT, "test.csv", args.test))
    corpus = base.find_file(base.ROOT, "corpus.parquet", args.corpus)
    archive = Archive(pd.read_parquet(corpus))
    test_ids, test_x = collect(archive, test, args.output, "test")
    if args.model:
        model = joblib.load(args.model)
    else:
        tpath = base.find_file(base.ROOT, "train.csv", args.train)
        train = base.read_questions(tpath)
        folds, splits = base.group_folds(train, base.find_file(base.ROOT, "train_groups.csv", args.groups))
        scorer = base.module_from_file(base.find_file(base.ROOT, "score.py"))
        gold = {q["id"]: q for q in scorer.load_gold([str(tpath)])}
        ids, features = collect(archive, train, args.output, "train")
        oof = [None] * len(train)
        for fold, (tr, va) in enumerate(splits):
            model = ranker(features, ids, train, gold, tr, args.threads)
            for i, r in zip(va, apply(model, train, ids, features, va, args.threads)):
                oof[i] = r
            print(f"OOF fold {fold + 1}/5", flush=True)
        dump_jsonl(args.output / "train.jsonl", oof)
        model = ranker(features, ids, train, gold, np.arange(len(train)), args.threads)
        joblib.dump(model, args.output / "model.joblib")
        seed = [json.loads(x) for x in (args.output / "seed_train.jsonl").read_text().splitlines()]
        def evaluate(rows):
            eligible = [(r, gold[r["qid"]]) for r in rows if gold[r["qid"]]["_G"]]
            result = {}
            for level in ("all", "light", "medium", "heavy"):
                samples = [(r, g) for r, g in eligible if level == "all" or g["level"] == level]
                weights = [g["_w"] for _, g in samples]
                ap = [scorer.ap5(r["ids"][:5], g["_G"]) for r, g in samples]
                result[level] = {"n": len(samples), "ap5": np.mean(ap), "weighted_ap5": np.average(ap, weights=weights)}
                for k in (5, 20, 100):
                    recalls = [len(set(r["ids"][:k]) & set(g["_G"])) / len(set(g["_G"])) for r, g in samples]
                    result[level][f"recall{k}"] = np.mean(recalls)
                    result[level][f"hit{k}"] = np.mean(np.asarray(recalls) > 0)
            return result
        report = {"notes": "Evidence only, nonempty gold. Five GroupKFold folds for learned model.",
                  "seed": evaluate(seed), "ranker": evaluate(oof)}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(report["ranker"]["all"], flush=True)
    dump_jsonl(args.output / "test.jsonl", apply(model, test, test_ids, test_x, range(len(test)), args.threads))
    print(f"Completed in {time.perf_counter() - begin:.1f}s", flush=True)


if __name__ == "__main__":
    main()

