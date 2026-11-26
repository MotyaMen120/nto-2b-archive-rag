"""Offline extractive reader for choice, yes/no and status questions."""

from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
import json
import math
import os
import re

import numpy as np
from nltk.stem.snowball import RussianStemmer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


STEMMER = RussianStemmer()
WORD = re.compile(r"\d+(?:[.,:]\d+)*|[а-яa-z]+")
STOP = set("в на из по у с за к для от до о об и или а но ни ли как было была был были что это эта этот те тот та тот же там наш наши нового нижнем нижний нижнего нижегородский области года году год который которую которые уже сейчас правда".split())
SMALL_NUMBERS = {
    "один": 1, "одна": 1, "одно": 1, "одной": 1, "одну": 1, "две": 2,
    "два": 2, "двух": 2, "двое": 2, "три": 3, "трех": 3, "трое": 3,
    "четыре": 4, "четырех": 4, "четверо": 4, "пять": 5, "пяти": 5,
    "пятеро": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8, "девять": 9, "десять": 10, "десяти": 10,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "сто": 100, "сотню": 100,
}
MONTH = ("январ", "феврал", "март", "апрел", "ма[йяею]", "июн",
         "июл", "август", "сентябр", "октябр", "ноябр", "декабр")
MONTH_RX = re.compile(r"\b(?:(\d{1,2})\s+)?(" + "|".join(f"(?:{m})[а-я]*" for m in MONTH) + r")(?:\s+(20\d{2}))?")
FUTURE = re.compile(r"\b(?:планир|запланир|собира|пообещ|обеща|намерен|проект|анонсир|рассчитыва|смож|будет|будут|собер|предстоит|должн|ожида|предлага|подготовят|планах|решено|собираются|намечен|могут)\w*|\b(?:откро|запуст|пуст|разреш|возобнов|заработа|состо|заверш|прим|введ|провед|отмен|закро|постро|начн|верн|зажг|восстанов)(?:ют|ет|ут|ется|ится|ится|ят|ат|ит|утся|нут|ны|ено|уться)\b")
POSTPONED = re.compile(r"\b(?:отложил|отложен|перенесл|перенесен|переносит|сдвинул|сдвинут|продлили срок|отсроч|не успел|позднее первоначаль)\w*")
CANCELLED = re.compile(r"\b(?:отменил|отменен|отменя[ею]|отказал[иа]сь|отказались|не состоится|не планиру|не будет|запрещен)\w*")
DONE = re.compile(r"\b(?:открыл|открылся|открыли|открылась|открыты|открыт[ао]?|начал|стартовал|приступил|пустил|запустил|сдали|состоял|прошл[аио]|провели|завершил|завершен|закончил|заключил|подписал|вступил|вышел|вышли|ввел|ввели|возобновил|возобновлен|вернули|вернулс|готов[ао]?|заработал|установил|построил|построен|обустроил|принял|принят|приняли|потушил|ликвидирова|вернулась|отпразднова)\w*")
NEGATION = re.compile(r"\b(?:не|нет|нельзя|запрещен\w*|лишен\w*|без|отсутств\w*)\b")

# The two sides refer to availability, service, work, or eligibility. This
# lets a post about an interruption answer a question phrased as permission.
FAMILIES = [
    (r"откры|пустил|пустят|ездить|ехать|проех|проезд|ходят|ходил|вернул|возит|возили|возил|возобнов|попасть|работа|прийти|начал|состоя|провел|купать|приним|получ|разреш|действов|доступ|горит|горел|увидеть|подсвеч|увидеть|свети|лететь|улететь|полет|летал|шла|стройк|заняти|обслужива|ездят",
     r"откр|пустил|проезд разреш|снова ход|возобнов|заработ|вернул|доступ|разреш|начал|приступ|состо|провел|продолжа|принима|начн|начат|открыти|старт|восстанов|возвращ|первый рейс|действу|включ|зажг|горит|горел|летает|отправл|появи|обслужива",
     r"перекры|закр|прекрат|приостанов|отмен|недоступ|не работа|огранич|времен.*не|нельзя|запрет|не будет|объезд|остановлен|погас|закрас|счист|стерл|сняли|убрал|вырез|сгор"),
    (r"останов|закры|перекры|прекрат|отмен|перестал|отказыва|запрет|не приним|отказ",
     r"перекры|закр|прекрат|приостанов|отмен|отказ|временно.*не|огранич|перестан|запрет",
     r"возобнов|откры|пустил|сохрани|продолжа|продлил|разреш|можно"),
    (r"отоплен|тепло|батаре",
     r"дали тепло|подал.*тепл|начал.*отопит|отопит.*старт|топить|подключ|включ|возобнов.*[отт]оп|тепло.*верн|возвращ.*тепл",
     r"отключ|заверш.*отопит|отопит.*заверш|выключ|прекрат|батареи.*холод|батареи.*осты"),
    (r"достро|заверш|закончи|додел|готов|довел|довели|потуш|закончи",
     r"заверш|законч|достро|сдал|сдан|готов|ликвидирова|потуш|заверши|ввел|ввели",
     r"продолжа|недостро|идут работ|ведутся работ|приступ|начал|планир|еще идет|еще не"),
    (r"побед|обыгра|выигра|выше|обошл", r"выигра|побед|обыгра|разгромил|одержал", r"проигра|уступил|поражени"),
    (r"уступ|проигра|поражени", r"проигра|уступ|поражени", r"побед|выигра|обыгра|разгромил"),
    (r"подпис|принял|принят|вступил|официально|закон|заплат|положен|выда|выплат",
     r"утвержден|утвердил|принял|принят|приняли|вступил|подписал|подписан|начал.*выплат|получил|действует|оформить",
     r"проект|предлож|планир|рассмотр|первом чтении|подготов"),
]
FAMILIES = [(re.compile(q), re.compile(y), re.compile(n)) for q, y, n in FAMILIES]


def clean(text):
    return str(text or "").lower().replace("ё", "е")


@lru_cache(maxsize=60000)
def stem(word):
    if word in SMALL_NUMBERS:
        return str(SMALL_NUMBERS[word])
    if word[:1].isdigit():
        return word.replace(",", ".")
    return STEMMER.stem(word)


@lru_cache(maxsize=18000)
def words(text):
    return frozenset(stem(word) for word in WORD.findall(text) if word not in STOP)


@lru_cache(maxsize=6000)
def sentences(text):
    return tuple(s.strip() for s in re.split(r"\n+|(?<=[!?])\s+|(?<=[.])\s+(?=[а-яa-z«])|[;]", clean(text)) if s.strip())


def explicit_date(text, reference):
    matches = list(MONTH_RX.finditer(text))
    if not matches:
        return None
    m = next((m for m in matches if m[1]), matches[0])
    month = next(index for index, pat in enumerate(MONTH, 1) if re.match(pat, m[2]))
    year = int(m[3]) if m[3] else reference.year - int(month > reference.month)
    before = text[max(0, m.start() - 16):m.start()]
    day = int(m[1]) if m[1] else 5 if "начал" in before else 25 if "конц" in before else 15
    try:
        return date(year, month, day)
    except ValueError:
        return None


def iso(value):
    return date.fromisoformat(str(value)[:10])


def content_numbers(text):
    """Keep assertions about a quantity, stripping its temporal setting."""
    stripped = MONTH_RX.sub(" ", text)
    stripped = re.sub(r"\b20\d{2}(?:-\w+)?\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", " ", stripped)
    stripped = re.sub(r"(?<=\d)\s+(?=\d{3}(?:\D|$))", "", stripped)
    vals = []
    for token in WORD.findall(stripped):
        if token in SMALL_NUMBERS:
            vals.append(str(SMALL_NUMBERS[token]))
        elif token[:1].isdigit():
            vals.append(token.replace(",", "."))
    return vals


def question_features(q, fmt):
    out = {}
    patterns = {
        "verification": r"правда|верно ли", "past": r"было|были|была|шл[аи]|можно ли было",
        "current": r"уже|сейчас|до сих пор|снова|пока", "plan": r"план|обещ|собира|анонс",
        "comparison": r"больше|меньше|выше|ниже|не менее|свыше|менее|более",
        "unchanged": r"по-прежнему|только|так же|вс[ею] еще",
        "money": r"рубл|стоим|билет|цен|стои[лт]|миллион|тысяч",
        "historical": r"(?:начале|середине|конце)|\d{1,2}\s+(?:январ|феврал|март|апрел|мая|июн|июл|август|сентябр|октябр|ноябр|декабр)",
        "sport": r"побед|выигр|проигр|обыгра|счет|матч|пенальти|шайб",
    }
    for name, pattern in patterns.items():
        out["q_" + name] = float(bool(re.search(pattern, q)))
    for j, (trigger, _, _) in enumerate(FAMILIES):
        out[f"family_{j}"] = float(bool(trigger.search(q)))
    out["q_negative"] = float(bool(NEGATION.search(q)))
    return out


def temporal_weight(age):
    return 1.0 / (1.0 + abs(age) / 90.0)


def entailment_features(path, row, posts):
    from cpu_nli_2b import compute
    visible = [(str(p["date"])[:10], str(p["text"])) for p in posts[:6]
               if str(p["date"])[:10] <= row["asked_at"]]
    return compute(path, row["question"], row["answer_format"], read_options(row.get("options")), visible)


def level_features(row, posts, nli_path=None):
    q = clean(row["question"])
    fmt = row["answer_format"]
    when = iso(row["asked_at"])
    target = explicit_date(q, when)
    historical = target is not None and bool(re.search(r"можно ли было|можно было|был[аио]? ли|шл[аи] ли|действовал|ходили|горел|работал|летал|шла ли", q))
    moment = target if historical else when
    q_words = words(q)
    q_nums = set(content_numbers(q))
    semantic = [(yes, no) for trig, yes, no in FAMILIES if trig.search(q)]
    out = question_features(q, fmt)
    out["q_num"] = min(3, len(q_nums)) / 3
    out["q_score"] = float(any(":" in num for num in q_nums))
    out["historical"] = float(historical)
    rows = []
    for i, post in enumerate(posts[:12]):
        pub = iso(post["date"])
        if pub > when:
            continue
        r_weight = (1.0 + i / 2.0) ** -.7
        age = (moment - pub).days
        for sentence in sentences(str(post["text"]))[:30]:
            tokens = words(sentence)
            lexical = len(q_words & tokens) / max(1, len(q_words))
            topic = r_weight * (.30 + min(.70, lexical * 2))
            numeric = set(content_numbers(sentence))
            future, past = bool(FUTURE.search(sentence)), bool(DONE.search(sentence))
            if any("не " + word in sentence for word in ("закры", "откры", "заверш", "приня", "нач", "прош")):
                past = False
            yes = sum(bool(p.search(sentence)) for p, _ in semantic) / max(1, len(semantic))
            no = sum(bool(p.search(sentence)) for _, p in semantic) / max(1, len(semantic))
            if bool(re.search(r"\bне\s+(?:работа|откры|доступ|горел|горят|будет|пуска|езд|ход|нач|приним|получ|проводи)", sentence)):
                yes, no = 0.0, max(no, .8)
            if historical and age < 0:
                sentence_date = explicit_date(sentence, pub)
                if sentence_date and sentence_date <= moment:
                    age_factor = temporal_weight((moment - sentence_date).days)
                else:
                    age_factor = .20 * temporal_weight(age)
            else:
                age_factor = temporal_weight(age)
            t = topic * age_factor
            events = {
                "future": float(future), "done": float(past), "postponed": float(bool(POSTPONED.search(sentence))),
                "cancelled": float(bool(CANCELLED.search(sentence))), "negative": float(bool(NEGATION.search(sentence))),
                "yes_event": yes, "no_event": no, "yes_fact": yes * (1 - .65 * future),
                "no_fact": no * (1 - .65 * future), "yes_plan": yes * future, "no_plan": no * future,
                "verified_event": yes * past, "verified_reverse": no * past, "has_number": float(bool(q_nums and numeric)),
                "num_match": len(q_nums & numeric) / max(1, len(q_nums)),
            }
            if q_nums and numeric:
                q_val = [float(n) for n in q_nums if ":" not in n and n.count(".") <= 1]
                p_val = [float(n) for n in numeric if ":" not in n and n.count(".") <= 1]
                comparable = [(a,b) for a in q_val for b in p_val if a and .45 < b/a < 2.2]
                if re.search(r"меньше|ниже|менее|не достиг", q):
                    events["bound"] = float(any(b < a for a,b in comparable))
                elif re.search(r"больше|выше|свыше|более|превыс", q):
                    events["bound"] = float(any(b > a for a,b in comparable))
            rows.append((topic, t, events))
    for name in ("future", "done", "postponed", "cancelled", "negative", "yes_event", "no_event",
                 "yes_fact", "no_fact", "yes_plan", "no_plan", "verified_event", "verified_reverse",
                 "num_match", "has_number", "bound"):
        values = [weight * evt.get(name, 0.) for weight, _, evt in rows]
        timed = [weight * evt.get(name, 0.) for _, weight, evt in rows]
        out[name] = max(values, default=0.)
        out["near_" + name] = max(timed, default=0.)
        out["avg_" + name] = sum(sorted(values, reverse=True)[:4]) / 4
    if rows:
        out["topicality"] = max(r[0] for r in rows)
    out["num_advantage"] = out["num_match"] - .55 * out["has_number"] if q_nums else 0.
    out["event_direction"] = out["near_yes_fact"] - out["near_no_fact"]
    out["realization"] = out["near_done"] - out["near_future"]
    if nli_path:
        for i, feats in enumerate(entailment_features(nli_path, row, posts)):
            out.update({f"{i}_{key}": value for key, value in feats.items()})
    return out


def read_options(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = []
    return [re.sub(r"^\s*[a-dа-г][.)]\s*", "", clean(s)) for s in (raw or [])][:4]


def choice_features(row, posts, nli_path=None):
    options = read_options(row.get("options"))
    q = clean(row["question"])
    when = iso(row["asked_at"])
    all_terms = [words(opt) for opt in options]
    frequency = Counter(term for terms in all_terms for term in terms)
    q_words = words(q)
    recency = bool(re.search(r"последн|на сегодня|сейчас|нынеш|по свеж", q))
    parts = []
    for i, post in enumerate(posts[:12]):
        pub = iso(post["date"])
        if pub > when:
            continue
        distance = (when - pub).days
        # Preserve a path for older posts when recency is irrelevant.
        time = .35 + .65 * temporal_weight(distance) if recency else 1.
        for s in sentences(str(post["text"]))[:30]:
            sw = words(s)
            topicality = len(q_words & sw) / max(1, len(q_words))
            weight = (1. + i / 3) ** -.7 * time * (.7 + min(.3, topicality))
            parts.append((s, sw, weight))
    features = []
    for opt, terms in zip(options, all_terms):
        numerals = set(content_numbers(opt))
        correction = bool(re.search(r"предпосылка неверна|вопрос невер|в действительности|на самом деле", opt))
        f = {"correction": int(correction), "negative_option": int(bool(NEGATION.search(opt))),
             "explanation": int(":" in opt and not re.search(r"\d:\d", opt)),
             "length": math.log1p(len(terms))/3, "number_option": int(bool(numerals)),
             "equal": int(bool(re.search(r"одинаков|поровну|равны|ни в одной|в обоих",opt)))}
        w = {t: math.log(1 + len(options) / frequency[t]) * (.8 if t in q_words else 1.) for t in terms}
        denom = max(sum(w.values()), 1e-8)
        values = defaultdict(list)
        for s, sw, weight in parts:
            inter = sw & terms
            overlap = sum(w[t] for t in inter) / denom
            nums = set(content_numbers(s)) if numerals else set()
            nm = len(numerals & nums)/max(1,len(numerals))
            rev_score = any(":" in n and ":".join(reversed(n.split(":"))) in nums for n in numerals)
            exact = len(opt) >= 3 and opt in s
            victory = re.search(r"побед|выигр|обыгра", opt)
            defeat = re.search(r"поражен|проигра|уступил", opt)
            polarity = bool(victory and re.search(r"выигра|обыгра|побед|разгромил",s) or
                            defeat and re.search(r"уступил|проигра|поражен",s))
            scores = {"overlap": overlap, "num": nm, "reverse": .65*rev_score,
                      "exact": float(exact), "score_and_side": float(polarity)*max(nm,.70*rev_score),
                      "joint": max(.60*overlap+.40*nm if numerals else overlap, .85*exact)}
            for key, val in scores.items():
                values[key].append(weight*val)
        for key, vals in values.items():
            f[key] = max(vals,default=0.)
            f["avg_"+key] = sum(sorted(vals,reverse=True)[:4])/4.
        features.append(f)
    relative = ["length","joint","overlap","avg_joint","num","exact","score_and_side"]
    if nli_path:
        for f, new in zip(features, entailment_features(nli_path, row, posts)):
            f.update(new)
        relative.extend(("nli_mean_entailment", "nli_max_entailment", "nli_top_entailment"))
    if features:
        for key in relative:
            avg = sum(f.get(key,0.) for f in features)/len(features)
            for f in features:
                f["relative_"+key] = f.get(key,0.) - avg
    return features


def _trained(features, targets, weights):
    if not targets or len(set(targets)) < 2:
        return targets[0] if targets else None
    vec = DictVectorizer(sparse=False)
    x = vec.fit_transform(features)
    linear = LogisticRegression(C=2.0, max_iter=350).fit(x, targets, sample_weight=weights)
    forest = ExtraTreesClassifier(n_estimators=90, min_samples_leaf=3, max_features=.75,
                                  random_state=901, n_jobs=1).fit(x, targets, sample_weight=weights)
    return vec, linear, forest


def probabilities(classifier, features):
    if classifier is None or isinstance(classifier, str):
        return [{classifier: 1.} for _ in features]
    vec, linear, forest = classifier
    x = vec.transform(features)
    proba = .55 * linear.predict_proba(x) + .45 * forest.predict_proba(x)
    return [dict(zip(linear.classes_, values)) for values in proba]


def fold_gold(row, corpus):
    found = []
    for key in str(row.get("evidence","")).split():
        value = corpus.get(key)
        if value is None and key.isdigit():
            value = corpus.get(int(key))
        if value is not None and str(value["date"])[:10] <= row["asked_at"]:
            found.append(value)
    return found


def fit(rows, corpus, ranked=None, nli_path=None):
    """Fit on outer-training questions only. ranked must be label-free or nested."""
    nli_path = nli_path or os.environ.get("NTO2B_NLI_DIR")
    buckets = {fmt: ([], [], []) for fmt in ("choice","status","yesno")}
    repeated = Counter(r["question"] for r in rows)
    for row in rows:
        fmt = row["answer_format"]
        if fmt not in buckets:
            continue
        gold = fold_gold(row, corpus)
        available = ranked.get(row["qid"], []) if ranked is not None else []
        contexts = [(gold, .55), (available, 1.)] if available else [(gold, 1.)]
        for posts, part_weight in contexts:
            fs, ys, ws = buckets[fmt]
            weight = part_weight / math.sqrt(repeated[row["question"]])
            if fmt == "choice":
                options = choice_features(row, posts, nli_path)
                for i, f in enumerate(options):
                    fs.append(f)
                    ys.append("yes" if "abcd"[i] == row["answer"] else "no")
                    ws.append(weight)
            else:
                fs.append(level_features(row, posts, nli_path))
                ys.append(row["answer"])
                ws.append(weight)
    model = {fmt: _trained(*bucket) for fmt,bucket in buckets.items()}
    if nli_path:
        model["nli_path"] = str(nli_path)
    return model


def predict(model, row, posts):
    fmt = row["answer_format"]
    nli_path = (os.environ.get("NTO2B_NLI_DIR") or model.get("nli_path")) if "nli_path" in model else None
    if fmt == "choice":
        fs = choice_features(row, posts, nli_path)
        if not fs:
            return "skip"
        probs = probabilities(model[fmt], fs)
        scores = [p.get("yes",0.) + .09*f.get("relative_joint",0.) for p,f in zip(probs,fs)]
        return "abcd"[int(np.argmax(scores))]
    f = level_features(row, posts, nli_path)
    probs = probabilities(model[fmt],[f])[0]
    result = max(probs,key=probs.get)
    return result or ("no" if fmt == "yesno" else "promised")
