"""Extractive CPU reader for Russian archive questions about dates and quantities."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache


def _norm(value):
    return str(value).lower().replace("ё", "е").replace("\xa0", " ").replace("\u202f", " ")


MONTH_PATTERN = r"январ\w*|феврал\w*|мар[тс]\w*|апрел\w*|майск\w*|ма[йяею]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*"
MONTH = re.compile(r"\b(?:" + MONTH_PATTERN + r")\b")
MONTH_PREFIX = "янв фев мар апр ма июн июл авг сен окт ноя дек".split()
DAY_MONTH = re.compile(
    r"(?<!\d)(?:(\d{1,2})(?:-го)?\s*(?:[-–—]|по|и)\s*)?(\d{1,2})(?:-го)?\s+("
    + MONTH_PATTERN + r")(?:\s+(19\d{2}|20\d{2}))?\b"
)
DIGITAL_DATE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(20\d{2}|\d{2})\b|\b(20\d{2})-(\d{2})-(\d{2})\b")
RELATIVE = {"сегодня": 0, "вчера": -1, "позавчера": -2, "завтра": 1, "послезавтра": 2}
STOP = set(("сколько какой какого какую каком какому какая какие когда чем это что если от до "
            "сейчас около примерно который которой котором которых которую после перед через просто "
            "составил составила сообщили сообщение новости начало конце день года году этом была было для "
            "ему его него том между прежде числе первом первой данные данным количество").split())
WORD = re.compile(r"[а-яa-z]+|\d+")
STEM_END = re.compile(r"(?:иями|ями|ами|ого|его|ыми|ими|ему|ому|иях|ать|ять|ить|ают|яют|или|али|ели|ий|ой|ей|ая|яя|ое|ые|ые|их|ых|ом|ем|ам|ям|ах|ях|ов|ев|ью|ию|ию|ая|а|у|я|и|ы|е|о|ю)$")


@lru_cache(maxsize=65000)
def _stem(word):
    return STEM_END.sub("", word)[:7] if len(word) > 4 else word


def _words(text):
    return {_stem(w) for w in WORD.findall(text) if (len(w) >= 3 or w.isdigit()) and w not in STOP}


def _month(word):
    return next(i for i, prefix in enumerate(MONTH_PREFIX, 1) if word.startswith(prefix))


@dataclass(frozen=True)
class Moment:
    value: date
    start: int
    end: int
    role: str


def _dates(text, pub):
    out = []
    for m in DIGITAL_DATE.finditer(text):
        a, b, c, y, mm, dd = m.groups()
        if a:
            y, mm, dd = (2000 + int(c) if len(c) == 2 else int(c)), b, a
        try:
            out.append(Moment(date(int(y), int(mm), int(dd)), *m.span(), "text"))
        except ValueError:
            continue
    for m in DAY_MONTH.finditer(text):
        first, last, word, year = m.groups()
        month = _month(word)
        # News in December and January frequently announces the adjacent year.
        implicit_year = pub.year + (pub.month >= 10 and month <= 2) - (pub.month <= 2 and month >= 10)
        for group, day, role in [(1, first, "start"), (2, last, "end" if first else "text")]:
            if day:
                try:
                    when = date(int(year or implicit_year), month, int(day))
                except ValueError:
                    continue
                out.append(Moment(when, m.start(group), m.end(), role))
    for m in re.finditer(r"\b(?:позавчера|послезавтра|вчера|сегодня|завтра)\b", text):
        out.append(Moment(pub + timedelta(days=RELATIVE[m[0]]), *m.span(), "relative"))
    return out


NUM_WORDS = {}
for _n, _forms in enumerate((
    "ноль", "один одна одно одного одну", "два две двух двое двоих двум", "три трех трое троих",
    "четыре четырех четверо", "пять пяти пятеро", "шесть шести шестеро", "семь семи", "восемь восьми",
    "девять девяти", "десять десяти", "одиннадцать", "двенадцать", "тринадцать",
    "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"
)):
    NUM_WORDS.update(dict.fromkeys(_forms.split(), _n))
NUM_WORDS.update({"полтора": 1.5, "полторы": 1.5, "двадцать": 20, "тридцать": 30, "сорок": 40,
                  "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80,
                  "девяносто": 90, "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
                  "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900})
ORDINALS = {"втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9, "десят": 10}
NUM_TOKEN = re.compile(r"(?<![\w\d])[-−]?\d+(?: \d{3}(?!\d))*(?:[.,]\d+)?(?!\d)|\b(?:"
                       + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r")\b")
SCALE = re.compile(r"\s*(тыс\.(?=\s|$)|тыс\b|тысяч\w*|млн\.?|миллион\w*|млрд\.?|миллиард\w*)")
ORDINAL = re.compile(r"\b(" + "|".join(ORDINALS) + r")\w{0,4}\s+(?:раз|мест\w*)\b")


def _scale(s):
    if s.startswith(("млрд", "миллиард")):
        return 1e9
    if s.startswith(("млн", "миллион")):
        return 1e6
    return 1e3 if s.startswith("тыс") else 1.0


UNIT_FORMS = {
    "руб": r"руб(?:\w*|\.)|₽", "%": r"%|процент\w*",
    "people": r"человек\w*|люд\w*|жител\w*|нижегородц\w*|пассажир\w*|участник\w*|зрител\w*|спортсмен\w*|болельщик\w*|пациент\w*|россиян\w*|бегун\w*|турист\w*|детей|ребен\w*|учащ\w*|гост\w*|девоч\w*|мальчи\w*|новорожден\w*|выпускник\w*|младен\w*|школьник\w*",
    "мест": r"мест\w*|ученик\w*|учащ\w*|зрител\w*|человек\w*|автомобил\w*",
    "место": r"мест\w*|строчк\w*|позици\w*", "лет": r"лет\w*|год\w*",
    "м": r"метр\w*|м\b", "км": r"километр\w*|км", "см": r"сантиметр\w*|см",
    "кв. м": r"квадрат\w*\s+метр\w*|кв\.?\s*м\w*|м[²2]",
    "раз": r"раз\w*", "м³": r"куб\w*\s+метр\w*|кубометр\w*|м[³3]",
    "месяцев": r"месяц\w*", "дней": r"дн\w*|день|суток|сутки",
    "минут": r"минут\w*|мин\b", "градусов": r"граду[сш]\w*|°",
    "медалей": r"медал\w*|наград\w*", "домов": r"дом\w*", "вызовов": r"вызов\w*",
}


@lru_cache(maxsize=256)
def _unit_spec(unit):
    scale = _scale(unit)
    base = re.sub(r"^(?:тыс\.?|тысяч\w*|млн|миллион\w*|млрд|миллиард\w*)\s+", "", unit)
    if base.startswith(("человек", "зрител", "болельщик")):
        base = "people"
    elif base in ("метров", "метра"):
        base = "м"
    elif base in ("м2", "м²"):
        base = "кв. м"
    elif base.startswith("градус"):
        base = "градусов"
    sample = re.sub(r"[^а-я0-9]+", " ", base).strip().split()
    pattern = UNIT_FORMS.get(base) or (re.escape(_stem(sample[-1])[:5]) + r"\w*" if sample else r"(?!)")
    return scale, re.compile("(?:" + pattern + ")"), base


@dataclass
class Document:
    text: str
    pub: date
    rank: int
    words: set
    moments: list


def _documents(row, posts, limit=20):
    cutoff = date.fromisoformat(str(row["asked_at"])[:10])
    docs = []
    for p in posts:
        pub = date.fromisoformat(str(p["date"])[:10])
        if pub > cutoff:
            continue
        text = _norm(p["text"])
        docs.append(Document(text, pub, len(docs), _words(text), _dates(text, pub)))
        if len(docs) >= limit:
            break
    return docs


def _context(text, start, end, width=125):
    left = max(text.rfind(mark, max(0, start - width), start) for mark in (". ", "!", "?", "\n"))
    right = min((p for mark in (". ", "!", "?", "\n")
                 if (p := text.find(mark, end, end + width)) >= 0), default=end + width)
    return text[max(start - width, left + 1, 0):right]


def _overlap(left, right):
    return len(left & right) / math.sqrt(max(1, len(left)))


def _rank_weight(rank):
    return 1 / (1 + .32 * rank) ** 1.35


def _number(v):
    return format(round(v, 6), ".6f").rstrip("0").rstrip(".") if v else "0"


@dataclass
class Mention:
    value: float
    start: int
    end: int
    strength: float
    hint: float
    shared: float
    positional: float
    word: bool


def _mentions(doc, qwords, focus, unit):
    scale, pattern, base = _unit_spec(unit)
    dates = [(m.start, m.end) for m in doc.moments]
    text = doc.text
    out = []
    for m in NUM_TOKEN.finditer(text):
        start, end = m.span()
        if any(a <= start < b for a, b in dates):
            continue
        raw = m[0]
        isword = raw in NUM_WORDS
        if isword:
            value = NUM_WORDS[raw]
        else:
            value = float(raw.replace(" ", "").replace(",", ".").replace("−", "-"))
        # Years, identifiers, addresses and times are usually context, not measurements.
        right = text[end:]
        if 1900 <= value <= 2099 and re.match(r"\s*(?:г[.\s]|год|[-–](?:го|м|й))", right):
            continue
        if text[max(0,start-1):start] in {"№", ":"} or re.match(r":\d{2}", right):
            continue
        mult = SCALE.match(right)
        value *= _scale(mult[1]) if mult else 1
        if mult:
            end += mult.end()
        if base == "руб" and (coins := re.match(r"\s*руб\w*\.?\s+(\d{1,2})\s*коп",text[end:])):
            value += int(coins[1])/100
        # Consecutive unit declarations typically follow the amount; inflected words
        # between an amount and its unit are common in count questions.
        following = text[end:end+52].lstrip(" .-–—)\t")
        preceding = text[max(0,start-36):start]
        direct = pattern.match(following)
        near = pattern.search(following)
        back = list(pattern.finditer(preceding))
        hint = 2.0 if direct else 1.0 if near and near.start() < 18 else .45 if back and len(preceding)-back[-1].end() < 14 else 0
        if base == "м" and following.startswith(("километр", "км")):
            value *= 1000
            hint = 2
        elif base == "км" and re.match(r"(?:метр\w*|м)\b", following):
            value /= 1000
            hint = 2
        elif base == "см" and re.match(r"(?:метр\w*|м)\b", following):
            value *= 100
            hint = 2
        elif base == "кв. м" and following.startswith(("га ", "гектар")):
            value *= 10000
            hint = 2
        ctx = _words(_context(text, start, end))
        shared = _overlap(qwords, ctx) + .75 * _overlap(focus, ctx)
        pos = 1 / (1 + start / 170)
        mismatch = bool(re.match(r"(?:рубл\w*|километр\w*|%|процент\w*|год\w*|человек\w*|метр\w*)\b", following)) and hint == 0
        strength = .75 + hint + .50 * min(4, shared) + .30 * pos - .70 * mismatch - .65 * (isword and hint == 0)
        out.append(Mention(value / scale,start,end,strength,hint,shared,pos,isword))
    if base in {"раз", "место"}:
        for m in ORDINAL.finditer(text):
            out.append(Mention(ORDINALS[m[1]],*m.span(),4,2,1,1,True))
    return out


@dataclass
class Hypothesis:
    answer: str
    strength: float
    rank: int
    unit: float = 0
    overlap: float = 0
    position: float = 0
    kind: str = "direct"


def _quantity(q, unit, docs):
    words = _words(q)
    focus = re.split(r"скольк\w*|како\w*", q, maxsplit=1)[-1]
    focus = _words(re.split(r"[,—?]|\b(?:в |по |что |котор)", focus, maxsplit=1)[0][:65])
    arithmetic = bool(re.search(r"на\s*сколько.*?(?:больш|меньш|дешев|дороже|подорож|вырос|возрос|сниз|теплее|прибав|подскоч)|во сколько раз|если вычесть|разниц|места вместе|наград.*вместе|сколько взрослых|среднесуточн|в среднем за одну|какую долю|двое взрослых", q))
    ops = []
    hypotheses = []
    for doc in docs:
        mentions = _mentions(doc, words, focus, unit)
        for m in mentions:
            strength = m.strength - .65 * bool(re.search(r"(?<!\d)" + re.escape(_number(m.value)) + r"(?!\d)", q))
            hypotheses.append(Hypothesis(_number(m.value),strength,doc.rank,m.hint,m.shared,m.positional))
        if arithmetic:
            operand_unit = unit
            if "%" in unit:
                operand_unit = "руб" if re.search(r"цен|стоимост|цен|рубл|дороже|дешев|подскоч|подорож|расход|деньги|сумм",q) else unit
            values = mentions if operand_unit == unit else _mentions(doc,words,focus,operand_unit)
            ops.extend((doc, m) for m in sorted(values,key=lambda x: -x.strength)[:7] if m.value > 0)
            if "среднесуточн" in q or "в среднем за одну" in q:
                for period in re.finditer(r"(\w+)\s+(?:дн|сут|встреч)",q):
                    divisor = int(period[1]) if period[1].isdigit() else NUM_WORDS.get(period[1],0)
                    if divisor:
                        for m in mentions:
                            if m.hint:
                                hypotheses.append(Hypothesis(_number(m.value/divisor),m.strength+1.5,
                                                             doc.rank,m.hint,m.shared,m.positional,"calc"))
    if arithmetic:
        pairs = sorted(ops, key=lambda p: -(p[1].strength * _rank_weight(p[0].rank)))[:18]
        for i, (a, x) in enumerate(pairs):
            for b, y in pairs[i+1:]:
                if x.value == y.value or (min(x.hint,y.hint) == 0 and not (a is b and max(x.hint,y.hint)>=1 and abs(x.start-y.start)<95)):
                    continue
                small, big = sorted((x.value,y.value))
                if "во сколько раз" in q:
                    value = big / small
                elif "двое взрослых" in q and "ребен" in q:
                    value = 2*big+small
                elif "%" in unit and "дол" in q:
                    value = small / big * 100
                elif "%" in unit and "процент" in q:
                    reducing = bool(re.search(r"дешев|сниз|упал|сократ|меньше",q))
                    value = (big - small) / (big if reducing else small) * 100
                elif "вместе" in q:
                    value = small + big
                else:
                    value = big - small
                proximity = .7 if a is b and abs(x.start-y.start) < 180 else 0
                if "до цел" in q:
                    value = round(value)
                if "до десятых" in q:
                    value = round(value,1)
                hypotheses.append(Hypothesis(_number(value), (x.strength+y.strength)/2+.70+proximity,
                                               max(a.rank,b.rank),min(x.hint,y.hint), (x.shared+y.shared)/2,0,"calc"))
    return hypotheses


FIRST_NEWS = re.compile(r"впервые|перв(?:ое|ая|ую) (?:новост|сообщен|публикац)")
PAST_EVENT = re.compile(r"открыли|открылся|открылась|пустили|начал[иа]? |прош[её]л|состоял|закрыли|возобновили|прекратили|отключили|заработал")
FUTURE_EVENT = re.compile(r"планиру|планов|запланир|планах|обеща|собира|откро[ею]|откроют|открыт[а-я]* в планах|пустят|заработает|долж[а-я]* (?:быть|начать|пройти)|начнут|закроют|состоится")


def _date_question(q, docs, cutoff):
    qwords = _words(q)
    first = bool(FIRST_NEWS.search(q))
    past = bool(PAST_EVENT.search(q)) and not re.search(r"обещ|планир|собира|долж|назнач", q)
    deadline = bool(re.search(r"до как|к как|по как|последн\w+ д", q))
    start_day = bool(re.search(r"с как|перв\w+ день|начал", q))
    qmonths = {_month(m[0]) for m in MONTH.finditer(q)}
    qyears = {int(y) for y in re.findall(r"\b20\d{2}\b", q)}
    hypotheses = []
    for doc in docs:
        relevant = _overlap(qwords, doc.words)
        nearby = sum(_rank_weight(other.rank) for other in docs if abs((doc.pub-other.pub).days) <= 2)
        if first:
            # Prefer the first report within a coherent wave of cross-channel posts.
            earlier = sum(_rank_weight(other.rank) for other in docs if 0 < (doc.pub-other.pub).days <= 3)
            later = sum(_rank_weight(other.rank) for other in docs if 0 < (other.pub-doc.pub).days <= 3)
            strength = 2.0 + min(2, relevant) * .65 + min(1.5, nearby)*.65 + min(1, later)*.75 - min(1, earlier)*1.20
            hypotheses.append(Hypothesis(doc.pub.isoformat(),strength,doc.rank,0,relevant,0,"first"))
            continue
        planned = len(FUTURE_EVENT.findall(doc.text))
        happened = len(PAST_EVENT.findall(doc.text))
        for m in doc.moments:
            if past and m.value > cutoff:
                if planned:
                    hypotheses.append(Hypothesis("promised",2.2 + min(1.5,relevant)*.6,doc.rank,1,relevant,0,"special"))
                continue
            loc = _words(_context(doc.text, m.start, m.end))
            overlap = _overlap(qwords, loc)
            before = doc.text[max(0,m.start-14):m.start]
            match = 1.3 * (deadline and (m.role == "end" or re.search(r"\b(?:до|по|к)\s*$",before) is not None))
            match += .95 * (start_day and (m.role == "start" or re.search(r"\bс\s*$",before) is not None))
            if m.role in ("start","end"):
                match -= .6 * ((deadline and m.role == "start") or (start_day and m.role == "end"))
            date_match = (.65 if m.value.month in qmonths else -.45) if qmonths else 0
            date_match += (.45 if m.value.year in qyears else -.55) if qyears else 0
            position = 1/(1+m.start/130)
            strength = 1.4 + min(3,relevant)*.40 + min(3,overlap)*.55 + match + date_match + position*.45
            hypotheses.append(Hypothesis(m.value.isoformat(),strength,doc.rank,match,overlap,position,"textdate"))
        if past or not doc.moments:
            strength = 1.0 + min(2,relevant)*.6 + min(2,happened)*.5 - min(2,planned)*.35
            hypotheses.append(Hypothesis(doc.pub.isoformat(),strength,doc.rank,0,relevant,0,"publication"))
        if past and planned and not happened:
            hypotheses.append(Hypothesis("promised",1.45+min(2,relevant)*.55,doc.rank,0,relevant,0,"special"))
    return hypotheses


EVENTS = [
    r"откр\w*|откры\w*|заработ\w*|пустил\w*|запуст\w*|возобнов\w*|приним\w* посетител\w*",
    r"закры\w*|перекры\w*|останов\w*|приостанов\w*|огранич\w*",
    r"отказ\w*|отмен\w*|не будет|не состоится",
    r"сдви\w*|перене\w*|перено\w*|отлож\w*",
    r"план\w*|обещ\w*|собир\w*|намер\w*|предлож\w*|хотят|хочет",
    r"законч\w*|заверш\w*|выполн\w*|готов\w*",
    r"нача\w*|старт\w*|приступ\w*",
    r"поддерж\w*|одобр\w*|приня\w*|проголос\w*",
    r"состо\w*|прош(?:ел|ла|ло|ли)|провели",
    r"увол\w*|расторг\w*|отстав\w*",
    r"загор\w*|вспых\w*|возгора\w*|пожар\w*",
    r"потуш\w*|ликвидир\w*|устран\w*",
]
EVENTS = [re.compile(r"\b(?:" + ev + ")") for ev in EVENTS]
NEWS = re.compile(r"новост|сообщен|объявлен|обещан|извести|слов |момент|одобрен|рассказ|публикац")


def _clauses(q):
    for p in (r"\bот\s+(.+?)\s+до\s+(?=(?:дня|новости|известия|его|ее|открытия|первого сообщения)\b)(.+)",r"\bот\s+(.+)\s+до\s+(.+)",r"\bмежду\s+(.+?)\s+и\s+((?:новост|сообщен|объявлен|днем|тем,|его |ее |извест|перв|датой).+)",
              r"\bмежду\s+(.+?)\s+и\s+(.+)",r"\bс\s+(?:момента,?\s*)?(.+)\s+до\s+(.+)",
              r"\bпосле\s+(.+?)\s+(?:сообщили|объявили|написали)\s+о\s+(.+)",
              r"\b(?:позже|позднее)\s+(.+?)[,?]\s*(.+)"):
        m = re.search(p,q)
        if m:
            return m.groups()
    return q, q


def _timeline_options(clause, other, docs):
    words = _words(clause)
    unique = words - _words(other)
    months = {_month(m[0]) for m in MONTH.finditer(clause)}
    events = [reg for reg in EVENTS if reg.search(clause) and not reg.search(other)]
    dates_in_story = defaultdict(set)
    for d in docs:
        dates_in_story[d.pub].add(d.rank)
    opts = []
    for doc in docs:
        semantic = .8 * _overlap(words,doc.words) + .8 * _overlap(unique,doc.words)
        semantic += .8 * sum(bool(reg.search(doc.text)) for reg in events)
        if months:
            semantic += 1.5 if doc.pub.month in months else -.55
        news = bool(NEWS.search(clause))
        explicit = bool(re.search(r"дат[уыой]|(?:объявленн\w+|названн\w+) срок",clause))
        support = sum(_rank_weight(rank) for rank in dates_in_story[doc.pub])
        base = semantic + 1.9 / (1+.18*doc.rank) + .40*min(2.0,support)
        opts.append((doc.pub,base + 1.5*news - .85*explicit,doc.rank,False))
        if not news or explicit:
            for m in doc.moments:
                if abs((m.value-doc.pub).days)>730:
                    continue
                context = _context(doc.text,m.start,m.end)
                loc = _overlap(unique,_words(context))
                s = base + .55*min(2,loc) + .6*sum(bool(e.search(context)) for e in events) + 1.4*explicit
                opts.append((m.value,s,doc.rank,True))
    # Limit pair enumeration and avoid giving long documents more votes for the same day.
    best = {}
    for dt, strength, rank, explicit in opts:
        key = (dt,rank)
        if key not in best or strength > best[key][1]:
            best[key] = (dt,strength,rank,explicit)
    return sorted(best.values(),key=lambda item:-item[1])[:16]


def _timeline(q, unit, docs, cutoff):
    first, second = _clauses(q)
    left = _timeline_options(first,second,docs)
    right = _timeline_options(second,first,docs)
    hypotheses = []
    for a, sa, ra, explicit_a in left:
        for b, sb, rb, explicit_b in right:
            if a >= b or b > cutoff and not explicit_b:
                continue
            if "месяц" in unit:
                n = (b.year-a.year)*12+b.month-a.month - (("полн" in q or "полн" in unit) and b.day<a.day)
            else:
                n = (b-a).days
            if n <= 0:
                continue
            strength = (sa+sb)/2 - .65*(ra==rb and not (explicit_a and explicit_b))
            hypotheses.append(Hypothesis(str(n),strength,min(ra,rb),float(explicit_a)+float(explicit_b),
                                          min(sa,sb),1/(1+max(ra,rb)),"timeline"))
    return hypotheses


def _is_timeline(q,unit):
    return ("дней" in unit or "месяц" in unit) and bool(re.search(
        r"прошл|позже|от .* до |между|с момент|сколько (?:дней|месяцев).*пока",q))


def _event_dates(clause, other, docs):
    """Describe each candidate date with information available at the time of the query."""
    own = _words(clause)
    distinctive = own-_words(other)
    months = {_month(m[0]) for m in MONTH.finditer(clause)}
    years = {int(y) for y in re.findall(r"\b20\d{2}\b",clause)}
    news = bool(NEWS.search(clause))
    explicit_day = bool(re.search(r"дат[уыой]|(?:названн\w+|объявленн\w+) срок|дня, когда|дня, с которого",clause))
    earliest = bool(re.search(r"перв|сам[а-я]+ ранн|изначальн",clause))
    events = [ev for ev in EVENTS if ev.search(clause) and not ev.search(other)]
    opposed = [ev for ev in EVENTS if ev.search(other) and not ev.search(clause)]
    support = Counter()
    for doc in docs:
        for shift in (-1,0,1):
            support[doc.pub+timedelta(days=shift)] += _rank_weight(doc.rank) / (1+abs(shift))
    endpoint = []
    for doc in docs:
        title_end = min(240, doc.text.find("\n") if "\n" in doc.text else len(doc.text))
        title = doc.text[:title_end]
        title_words = _words(title)
        head = _overlap(distinctive,title_words)
        full = _overlap(distinctive,doc.words)
        subject = _overlap(own,doc.words)
        matches = sum(bool(e.search(doc.text)) for e in events)
        opposite = sum(bool(e.search(title)) for e in opposed)
        lead_match = sum(bool(e.search(title)) for e in events)
        moments = [Moment(doc.pub,0,0,"pub"),*doc.moments]
        best = {}
        for m in moments:
            if abs((m.value-doc.pub).days) > 740:
                continue
            publication = m.role == "pub"
            local = title if publication else _context(doc.text,m.start,m.end)
            words = title_words if publication else _words(local)
            context = _overlap(distinctive,words)
            local_events = sum(bool(e.search(local)) for e in events)
            in_month = int(m.value.month in months) if months else 0
            in_year = int(m.value.year in years) if years else 0
            age = (m.value-doc.pub).days
            rank = doc.rank
            raw = (subject*.40 + full*.73 + head*.40 + matches*.50 + lead_match*.23 +
                   context*.35 + local_events*.55 - opposite*.25 + in_month*1.5 + in_year*.42 +
                   1.85/(1+.15*rank) + publication*(news*1.55-explicit_day*.9) +
                   (not publication)*explicit_day*.7 + min(1.2,support[doc.pub])*.40)
            vector = (raw,subject,full,head,context,matches,lead_match,opposite,local_events,
                      _rank_weight(rank),int(publication),int(publication and news),
                      int(not publication and explicit_day),int(earliest),int(news),float(in_month),
                      float(in_year),min(6,support[doc.pub]),math.copysign(math.log1p(abs(age)),age),
                      float("напомним" in local or "ранее" in local),float(bool(FUTURE_EVENT.search(local))))
            prev = best.get(m.value)
            if prev is None or prev[0] < raw:
                best[m.value] = (raw,m.value,doc.rank,vector)
        endpoint.extend(best.values())
    endpoint.sort(key=lambda x:-x[0])
    selected = []
    copies = Counter()
    for item in endpoint:
        if copies[item[1]] < 3:
            selected.append(item)
            copies[item[1]] += 1
        if len(selected) == 25:
            break
    return selected


@lru_cache(maxsize=180)
def _pair_samples(q,unit,cutoff,pairs):
    row = {"asked_at":cutoff}
    docs = _documents(row,({"date":dt,"text":text} for dt,text in pairs),limit=60)
    first,second = _clauses(q)
    left = _event_dates(first,second,docs)
    right = _event_dates(second,first,docs)
    out = []
    now = date.fromisoformat(cutoff[:10])
    for pos_a,(sa,a,ra,fa) in enumerate(left):
        for pos_b,(sb,b,rb,fb) in enumerate(right):
            if a>=b or b>now and fb[10]:
                continue
            if "месяц" in unit:
                n = (b.year-a.year)*12+b.month-a.month - (("полн" in q or "полн" in unit) and b.day<a.day)
            else:
                n = (b-a).days
            if n<=0:
                continue
            guess = (sa+sb)/2 - .5*(ra==rb and not (fa[10]==0 and fb[10]==0))
            summary = (guess,min(sa,sb),(fa[9]+fb[9])/2,min(fa[9],fb[9]),int(ra==rb),
                       float("месяц" in unit),math.log1p(n), math.log1p(max(0,(now-b).days)),
                       1/(1+pos_a),1/(1+pos_b),float(a==docs[ra].pub and b==docs[rb].pub))
            out.append((str(n),ra,rb,(*summary,*fa,*fb)))
    return out


def _learned_timeline(row,posts,model):
    import numpy as np
    from threadpoolctl import threadpool_limits
    pairs = tuple((str(p["date"])[:10],str(p["text"])) for p in posts[:60])
    samples = _pair_samples(_norm(row["question"]),_norm(row["unit"]),str(row["asked_at"])[:10],pairs)
    if not samples:
        return None
    x = np.asarray([s[3] for s in samples])
    with threadpool_limits(limits=1):
        logits = model.decision_function(x)
    by_value = defaultdict(list)
    for (value,_,_,features),logit in zip(samples,logits):
        by_value[value].append(float(logit) + .47*features[0])
    # Count independently supported date pairs with a cap, because many pairings
    # of reposts of the same two announcements can give the same duration.
    ranked = []
    for value, scores in by_value.items():
        top = sorted(scores,reverse=True)[:6]
        stacked = sum(math.exp((score-top[0])/1.2) for score in top)
        ranked.append((top[0]+.48*math.log(stacked),value))
    ranked.sort(reverse=True)
    margin = ranked[0][0]-ranked[1][0] if len(ranked)>1 else 1
    return ranked[0][1],1/(1+math.exp(.8-min(3,margin)))


KINDS = ("direct","calc","timeline","first","textdate","publication","special","count")


def _candidates(row, posts):
    q,unit = _norm(row["question"]),_norm(row.get("unit") or "")
    docs = _documents(row,posts)
    cutoff = date.fromisoformat(str(row["asked_at"])[:10])
    if row["answer_format"] == "date":
        hypotheses = _date_question(q,docs,cutoff)
    elif _is_timeline(q,unit):
        hypotheses = _timeline(q,unit,docs,cutoff)
    else:
        hypotheses = _quantity(q,unit,docs)
        if unit == "раз" and "во сколько" not in q and docs:
            dates = sorted({d.pub for d in docs[:7]})
            clusters = 1 + sum((b-a).days > 3 for a,b in zip(dates,dates[1:]))
            if re.search(r"перенос|сдвиг|откладыва",q):
                triggers = [d for d in docs[:12] if EVENTS[3].search(d.text)]
                if triggers:
                    dates = sorted({d.pub for d in triggers})
                    clusters = 1+sum((b-a).days>3 for a,b in zip(dates,dates[1:]))
            hypotheses.append(Hypothesis(str(clusters),2.8,0,0,0,0,"count"))
    by_value = defaultdict(dict)
    for h in hypotheses:
        # Repeated mentions in a single source must not outvote corroborating sources.
        key = h.rank
        previous = by_value[h.answer].get(key)
        if previous is None or previous.strength < h.strength:
            by_value[h.answer][key] = h
    features = {}
    for answer, by_rank in by_value.items():
        hs = list(by_rank.values())
        best = max(hs,key=lambda h: h.strength + .9*math.log(_rank_weight(h.rank)))
        strength = best.strength
        vote = sum(_rank_weight(h.rank)*min(2, math.exp(min(1,h.strength-strength))) for h in hs)
        pos = min(h.rank for h in hs)
        rule = strength + 1.20*math.log(_rank_weight(best.rank)) + .9*math.log1p(vote)
        vec = [rule,strength,vote,math.log1p(len(hs)),1/(1+pos),1/(1+best.rank),
               best.unit,best.overlap,best.position,sum(_rank_weight(h.rank) for h in hs),
               len(docs)/20, len(hs)/max(1,len(docs)), *[float(best.kind==k) for k in KINDS],
               float("руб" in unit),float("%" in unit),float("месяц" in unit)]
        if row["answer_format"] == "number" and answer not in ("skip","promised","no_info"):
            v = abs(float(answer))
            last_features = [math.log1p(v),float(v==int(v)),float(v<10),float(v>0 and v%10==0)]
        else:
            last_features = [0.0]*4
        features[answer] = vec + last_features
    if features:
        max_rule = max(vec[0] for vec in features.values())
        max_vote = max(vec[2] for vec in features.values())
        for vec in features.values():
            vec.extend((vec[0]-max_rule,vec[2]/(max_vote or 1)))
    return features


def _match(value, gold, fmt):
    if fmt != "number":
        return value==gold
    try:
        return abs(float(value)-float(gold)) <= .01*abs(float(gold))
    except ValueError:
        return value==gold


@lru_cache(maxsize=1000)
def _training_candidates(question, asked_at, fmt, unit, post_data):
    row = dict(question=question,asked_at=asked_at,answer_format=fmt,unit=unit)
    return _candidates(row,[{"date":dt,"text":txt} for dt,txt in post_data])


def _special_features(row,posts):
    q = _norm(row["question"])
    unit = _norm(row.get("unit") or "")
    qwords = _words(q)
    docs = _documents(row, posts)[:6]
    pattern = _unit_spec(unit)[1]
    when = date.fromisoformat(str(row["asked_at"])[:10])
    if not docs:
        return [0.0] * 17
    terms = [max(0,_overlap(qwords,d.words)) for d in docs]
    units = [bool(pattern.search(d.text)) for d in docs]
    plans = [bool(FUTURE_EVENT.search(d.text)) for d in docs]
    past = [bool(PAST_EVENT.search(d.text)) for d in docs]
    eligible_dates = [m.value for d in docs for m in d.moments]
    return [min(4,max(terms))/4,min(4,sum(terms)/len(terms))/4,
            sum(units)/len(units),float(units[0]),
            sum(plans)/len(plans), sum(past)/len(past),
            float(any(m>when for m in eligible_dates)),float("руб" in unit),
            float(bool(re.search(r"будет|обойдется|собира|планир",q))),
            float(bool(re.search(r"льгот|пенсион|студен|школьник|специальн",q))),
            float(bool(PAST_EVENT.search(q))),
            min(365,(when-docs[0].pub).days)/365,
            len({d.pub for d in docs})/len(docs),
            float(bool(FIRST_NEWS.search(q))),float("сколько сто" in q),
            float(bool(re.search(r"уже|сейчас",q))),min(30,len(qwords))/30]


def _lookup(posts_by_id, key):
    post = posts_by_id.get(int(key))
    return post if post is not None else posts_by_id.get(str(key))


def _model_data(row,pairs):
    return _training_candidates(str(row["question"]),str(row["asked_at"])[:10],row["answer_format"],
                                str(row.get("unit") or ""),tuple(pairs))


def fit_ranker(rows, posts_by_id, seed_rankings=None):
    """Fit only with *outer-training* labels and their temporally eligible evidence."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack,csr_matrix
    from threadpoolctl import threadpool_limits

    data = defaultdict(lambda: ([],[],[]))
    special = defaultdict(lambda: ([],[],[]))
    pair_x,pair_y,pair_w = [],[],[]
    for row in rows:
        fmt = row["answer_format"]
        if fmt not in {"number","date"}:
            continue
        gold = str(row["answer"]).strip()
        contexts = []
        pair_contexts = []
        gold_context = []
        for s in str(row["evidence"]).split():
            if not s.isdigit():
                continue
            p = _lookup(posts_by_id,s)
            if p is not None and str(p["date"])[:10] <= str(row["asked_at"])[:10]:
                gold_context.append((str(p["date"])[:10],str(p["text"])))
        if gold_context:
            contexts.append((gold_context,0.55 if seed_rankings is not None else 1.0))
            pair_contexts.append((gold_context,set(range(len(gold_context))),.55 if seed_rankings else 1))
        if seed_rankings is not None and row["qid"] in seed_rankings:
            ranking = seed_rankings[row["qid"]]
            is_time = _is_timeline(_norm(row["question"]),_norm(row.get("unit") or ""))
            ranked_pairs = [(int(k),_lookup(posts_by_id,k)) for k in ranking["ids"][:60 if is_time else 20]]
            ranked_pairs = [(k,p) for k,p in ranked_pairs if p is not None and str(p["date"])[:10]<=str(row["asked_at"])[:10]]
            ranked = [p for _,p in ranked_pairs]
            if ranked:
                as_pairs = [(str(p["date"])[:10],str(p["text"])) for p in ranked]
                contexts.append((as_pairs[:20],1.0))
                gold_ids = set(str(row["evidence"]).split())
                known_ranks = {i for i,(k,_) in enumerate(ranked_pairs) if str(k) in gold_ids}
                pair_contexts.append((as_pairs,known_ranks,1))
                texts,features,labels = special[fmt]
                texts.append(_norm(row["question"]))
                features.append(_special_features(row,ranked))
                labels.append(gold==("promised" if fmt=="date" else "no_info"))
        for pairs, context_weight in contexts:
            cand = _model_data(row,pairs)
            yes = [v for v in cand if _match(v,gold,fmt)]
            if not yes or len(yes)==len(cand):
                continue
            no = sorted((v for v in cand if v not in yes),key=lambda a:-cand[a][0])[:24]
            x,y,w = data[fmt]
            for vals, label, weight in [(yes,1,1.0),(no,0,2.0)]:
                for value in vals:
                    x.append(cand[value])
                    y.append(label)
                    w.append(context_weight*weight/len(vals))
        q,unit = _norm(row["question"]),_norm(row.get("unit") or "")
        if fmt=="number" and _is_timeline(q,unit):
            for pairs,known,weight in pair_contexts:
                samples = _pair_samples(q,unit,str(row["asked_at"])[:10],tuple(pairs))
                matches = [s for s in samples if _match(s[0],gold,fmt) and s[1] in known and s[2] in known]
                if not matches:
                    continue
                positives = sorted(matches,key=lambda s:-s[3][0])[:14]
                neg = sorted((s for s in samples if not _match(s[0],gold,fmt)),key=lambda s:-s[3][0])
                negatives = neg[:30]+neg[30::max(1,(len(neg)-30)//18)][:18]
                for subset,target,sw in [(positives,1,weight),(negatives,0,2*weight)]:
                    for sample in subset:
                        pair_x.append(sample[3])
                        pair_y.append(target)
                        pair_w.append(sw/len(subset))
    models = {}
    with threadpool_limits(limits=1):
        if sum(pair_y)>12 and len(pair_y)-sum(pair_y)>12:
            model = HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,learning_rate=.07,
                                                   min_samples_leaf=14,l2_regularization=1.5,
                                                   early_stopping=False,random_state=37)
            model.fit(pair_x,pair_y,sample_weight=pair_w)
            models["date_pairs"] = model
        for fmt,(x,y,w) in data.items():
            if sum(y) < 10:
                continue
            model = HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=7,l2_regularization=2,
                                                   min_samples_leaf=13,learning_rate=.07,random_state=37,
                                                   early_stopping=False)
            model.fit(x,y,sample_weight=w)
            models[fmt] = model
        for fmt,(texts,features,labels) in special.items():
            if min(sum(labels),len(labels)-sum(labels)) < 2:
                continue
            tokenizer = TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,
                                        sublinear_tf=True,max_features=9000)
            x = hstack([tokenizer.fit_transform(texts)*2.5,csr_matrix(features)],format="csr")
            model = LogisticRegression(C=2.0,solver="liblinear",class_weight={False:1.0, True:2.5},
                                       random_state=37, max_iter=150)
            model.fit(x,labels)
            models["special_"+fmt] = (tokenizer,model)
    return models


def predict_with_confidence(row, posts, ranker=None):
    """Return an answer and its heuristic confidence (use held-out calibration if needed)."""
    cand = _candidates(row,posts)
    if not cand:
        time_model = (ranker or {}).get("date_pairs")
        if time_model is not None and row["answer_format"]=="number" and _is_timeline(_norm(row["question"]),_norm(row.get("unit") or "")):
            result = _learned_timeline(row,posts,time_model)
            if result:
                return result
        return "skip",0.0
    vals = list(cand)
    scores = [cand[v][0] for v in vals]
    model = (ranker or {}).get(row["answer_format"])
    if model is not None:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            logits = model.decision_function([cand[v] for v in vals])
        scores = [r+.95*float(model_score) for r,model_score in zip(scores,logits)]
    order = sorted(range(len(vals)),key=lambda i:-scores[i])
    i = order[0]
    margin = scores[i] - scores[order[1]] if len(order)>1 else 1
    vote = cand[vals[i]][2]
    confidence = 1/(1+math.exp(-min(12, -.85+min(2.5,margin)+.45*math.log1p(vote))))
    special = (ranker or {}).get("special_"+row["answer_format"])
    if special is not None:
        from scipy.sparse import csr_matrix,hstack
        tokenizer,model = special
        x = hstack([tokenizer.transform([_norm(row["question"])])*2.5,csr_matrix([_special_features(row,posts)])])
        probability = model.predict_proba(x)[0,1]
        if probability > .5:
            return ("promised" if row["answer_format"]=="date" else "no_info"),float(probability)
    return vals[i],confidence


def predict(row, posts, ranker=None):
    return predict_with_confidence(row,posts,ranker)[0]
