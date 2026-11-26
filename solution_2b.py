#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
НТО ИИ 2026/27 — Командный этап | Кейс 2B «Спроси архив»
Модульное офлайн-решение с гибридным темпоральным поиском (BM25 + Dense MaxP),
словарём нижегородских перифраз, декомпозицией составных вопросов,
хронологической реконструкцией историй и оптимизацией доказательств под AP@5.

Аппаратные ограничения: 1x NVIDIA T4 (16 ГБ VRAM), 12 ГБ ОЗУ, <= 6 часов, 100% офлайн.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Принудительный офлайн-режим для трансформеров и хаба
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =============================================================================
# 1. КОНФИГУРАЦИЯ ПАЙПЛАЙНА
# =============================================================================

@dataclass
class PipelineConfig:
    data_dir: Path = Path(".")
    output_csv: Path = Path("submission.csv")
    cache_dir: Optional[Path] = None

    # Идентификаторы моделей (совместимы с офлайн-кэшем бейзлайна)
    encoder_id: str = "intfloat/multilingual-e5-small"
    encoder_revision: Optional[str] = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    reranker_id: Optional[str] = None  # Например, BAAI/bge-reranker-v2-m3 при наличии в кэше
    llm_id: str = "Qwen/Qwen3.5-2B"
    llm_revision: Optional[str] = "15852e8c16360a2fea060d615a32b45270f8a8fc"

    # Параметры индексации и чанкинга длинных постов
    enc_max_length: int = 384
    enc_batch_size: int = 64
    chunk_words: int = 210
    chunk_stride: int = 140
    max_chunks_per_post: int = 3

    # Параметры гибридного поиска
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    lex_pool: int = 160
    sem_pool: int = 160
    rrf_k: int = 50
    reader_top_k: int = 10
    evidence_max_k: int = 5

    # Параметры генерации LLM
    max_input_tokens: int = 4096
    max_new_tokens: int = 220

    # Пороги принятия решений с учётом штрафов метрики (aq: +1.0 / -0.25 skip / -0.5 wrong)
    no_info_max_lex_score: float = 2.2
    no_info_max_sem_score: float = 0.77
    cross_channel_dup_sim: float = 0.88


# =============================================================================
# 2. СЛОВАРЬ ПЕРИФРАЗ И ТОПОНИМОВ НИЖЕГОРОДСКОЙ ОБЛАСТИ
# =============================================================================

# Вопросы уровней medium/heavy часто используют перифразы вместо прямых названий.
# Словарь ниже связывает косвенные описания с каноническими терминами архива.
NIZHNY_PERIPHRASIS_RULES: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"станци[яиюей]\s+между\s+заречн\w+\s+и\s+пролетарск\w+", re.I),
        "метро Двигатель Революции спуск павильон крыша",
    ),
    (
        re.compile(r"(зел[её]н\w+\s+(зубаст\w+|хищник\w*|рептили\w*)|крокодил\w*\s+.*картин\w*)", re.I),
        "крокодил граффити стрит-арт подпорная стена Канавинский мост Похвалинский съезд",
    ),
    (
        re.compile(r"под\s+мостом\s+к\s+ярмарк\w+|у\s+канавинск\w+\s+мост\w+", re.I),
        "Канавинский мост подпорная стена Нижне-Волжская набережная",
    ),
    (
        re.compile(r"(городок\s+ножовщик\w+|под\s+павлов\w+|павловск\w+\s+автобус\w*\s+.*по\s+вод\w+)", re.I),
        "Ворсма озеро Тосканка плавающий автобус ПАЗ амфибия Павлово",
    ),
    (
        re.compile(r"(ажурн\w+\s+сетчат\w+\s+вышк\w+|стальн\w+\s+конструкц\w+\s+.*инженер\w+|в\s+городе\s+химиков)", re.I),
        "Шуховская башня Дзержинск берег Оки Владимир Шухов",
    ),
    (
        re.compile(r"главн\w+\s+областн\w+\s+больниц\w+\s+на\s+родионова", re.I),
        "НОКБ им. Семашко больница Семашко улица Родионова",
    ),
    (
        re.compile(r"(вагончик\w*\s+по\s+склон\w+\s+.*кремл\w+|предшественник\w*\s+к\s+выставк\w+\s+1896|подъ[её]мник\w*\s+у\s+стен\s+кремл\w+)", re.I),
        "кремлёвский фуникулёр элеватор Нижегородский кремль Северная башня Часовая башня",
    ),
    (
        re.compile(r"(канатк\w+|воздушн\w+\s+переправ\w+\s+на\s+бор|кабинк\w+\s+над\s+волгой)", re.I),
        "канатная дорога Нижний Новгород Бор кабинки проезд",
    ),
    (
        re.compile(r"(хоккейн\w+\s+двор[ец]\w*|арен\w+\s+у\s+слияния\s+рек)", re.I),
        "Ледовый дворец на Стрелке ХК Торпедо",
    ),
    (
        re.compile(r"(главн\w+\s+парк\w*\s+на\s+откос\w+|парк\w*\s+вдоль\s+проспекта\s+гагарина)", re.I),
        "парк Швейцария проспект Гагарина",
    ),
    (
        re.compile(r"(бело-сини\w+\s+корабл\w*\s+на\s+подводн\w+\s+крыль\w+|крылат\w+\s+суд\w+)", re.I),
        "Валдай Метеор СПК Водолёт речная навигация",
    ),
    (
        re.compile(r"(старичок\w*\s+с\s+деревянн\w+\s+салон\w+|ретро-?трамва\w+)", re.I),
        "ретро-трамвай РВЗ-6 КМ маршрут № 2 Городское кольцо",
    ),
]

STOPWORDS_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
    "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была",
    "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",
    "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас",
    "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два",
    "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между", "каком", "какие",
    "каких", "сколько", "какого", "какую", "какое", "году", "года", "год", "лет",
}


class NizhnyNovgorodQueryExpander:
    """Расширяет запросы синонимами, топонимами и декомпозирует вопросы-сравнения."""

    def __init__(self) -> None:
        self.learned_expansions: Dict[str, List[str]] = {}

    def fit_from_train(
        self,
        train_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        max_assoc_per_word: int = 3,
    ) -> None:
        """
        Автоматически находит ассоциации между словами из вопросов train.csv
        и характерными терминами из их золотых постов evidence.
        """
        if train_df is None or len(train_df) == 0:
            return
        post_text_map = dict(zip(posts_df["post_id"].astype(int), posts_df["text"].astype(str)))

        cooc: Dict[str, Counter] = {}
        q_freq: Counter = Counter()

        for _, row in train_df.iterrows():
            ev_raw = str(row.get("evidence", "") or "").strip()
            if not ev_raw:
                continue
            ev_ids = [int(x) for x in ev_raw.split() if x.isdigit() and int(x) in post_text_map]
            if not ev_ids:
                continue

            q_tokens = set(tokenize_stems(str(row["question"])))
            p_tokens: set[str] = set()
            for pid in ev_ids[:5]:
                p_tokens.update(tokenize_stems(post_text_map[pid]))

            diff_tokens = p_tokens - q_tokens
            for qt in q_tokens:
                if len(qt) < 4 or qt.isdigit():
                    continue
                q_freq[qt] += 1
                if qt not in cooc:
                    cooc[qt] = Counter()
                for pt in diff_tokens:
                    if len(pt) >= 4 and not pt.isdigit():
                        cooc[qt][pt] += 1

        for qt, cnt in cooc.items():
            if q_freq[qt] < 2:
                continue
            best = [
                pt for pt, c in cnt.most_common(max_assoc_per_word)
                if c >= max(2, int(0.6 * q_freq[qt]))
            ]
            if best:
                self.learned_expansions[qt] = best

    def expand_query(self, question: str, options: Optional[str] = None) -> List[str]:
        """
        Возвращает список поисковых подзапросов:
        1) основной расширенный запрос (вопрос + варианты ответа + раскрытые перифразы),
        2) дополнительные подзапросы, если вопрос сравнивает два объекта или два года (B2).
        """
        base_parts = [question.strip()]
        if options and str(options).strip() and str(options).lower() != "nan":
            # Убираем буквенные префиксы a), b), c), d) из вариантов выбора
            cleaned_opts = re.sub(r"\b[a-d]\)\s*", " ", str(options))
            base_parts.append(cleaned_opts)

        extra_terms: List[str] = []
        for pattern, expansion in NIZHNY_PERIPHRASIS_RULES:
            if pattern.search(question):
                extra_terms.append(expansion)

        if self.learned_expansions:
            q_stems = set(tokenize_stems(question))
            learned = []
            for st in q_stems:
                if st in self.learned_expansions:
                    learned.extend(self.learned_expansions[st])
            if learned:
                extra_terms.append(" ".join(learned[:8]))

        # Для вопросов о переносах / сроках добавляем маркеры изменений расписания
        q_low = question.lower()
        if any(w in q_low for w in ("откладывал", "переносил", "сдвинул", "впервые пообещал", "обещани")):
            extra_terms.append("перенесли срок откроют планируют завершить работы")

        main_query = " ".join(base_parts + extra_terms)
        subqueries = [main_query]

        # Декомпозиция вопросов-сравнений (B2): например, два года (2024 и 2025) или варианты a) vs b)
        years = sorted(set(re.findall(r"\b(202[0-6])\b", question)))
        if len(years) >= 2:
            q_without_years = re.sub(r"\b202[0-6]\b", " ", question)
            for y in years:
                subqueries.append(f"{q_without_years} {y} {' '.join(extra_terms)}")

        if options and str(options).strip() and str(options).lower() != "nan":
            opt_lines = [
                re.sub(r"^[a-d]\)\s*", "", line.strip())
                for line in str(options).splitlines()
                if line.strip()
            ]
            # Если в вариантах ответа перечислены разные объекты (короткие именные группы),
            # добавляем подзапрос по каждому объекту
            if 2 <= len(opt_lines) <= 4 and all(len(x.split()) <= 6 for x in opt_lines):
                for opt_text in opt_lines[:3]:
                    if not any(neg in opt_text.lower() for neg in ("не было", "не запускали", "одинаково")):
                        subqueries.append(f"{question} {opt_text}")

        return subqueries


# =============================================================================
# 3. ЛЕКСИЧЕСКИЙ И СЕМАНТИЧЕСКИЙ ПОИСК С ЧАНКИНГОМ (MaxP)
# =============================================================================

TOKEN_RE = re.compile(r"[а-яa-z]+|\d+", re.I)


def tokenize_stems(text: str) -> List[str]:
    """Нормализация и быстрый префиксный стемминг для русского языка."""
    raw = TOKEN_RE.findall(text.lower().replace("ё", "е"))
    stems: List[str] = []
    for w in raw:
        if w in STOPWORDS_RU:
            continue
        stems.append(w[:6] if len(w) >= 6 else w)
    return stems


class TemporalHybridRetriever:
    """
    Гибридный ретривер по архиву городских сообществ:
    - Индексирует посты с учётом названия канала (`channel`) и текста (`text`).
    - Разбивает длинные посты на перекрывающиеся чанки для dense-индекса (MaxP),
      чтобы факты в конце длинных сводок не обрезались лимитом токенов.
    - Строго фильтрует кандидатов по условию `post_date <= asked_at` (дата D).
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.posts_df: Optional[pd.DataFrame] = None
        self.post_ids: Optional[np.ndarray] = None
        self.post_dates: Optional[np.ndarray] = None
        self.post_channels: Optional[List[str]] = None

        # BM25 структуры
        self.inv_index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.idf: Dict[str, float] = {}
        self.doc_norm: Optional[np.ndarray] = None

        # Dense структуры (чанки -> родительский индекс поста)
        self.chunk_embs: Optional[np.ndarray] = None
        self.chunk_to_doc: Optional[np.ndarray] = None
        self.doc_primary_emb: Optional[np.ndarray] = None

        self.encoder_tokenizer = None
        self.encoder_model = None
        self.device = "cpu"

    def build_bm25(self, posts_df: pd.DataFrame) -> None:
        t0 = time.time()
        n_docs = len(posts_df)
        doc_len = np.zeros(n_docs, dtype=np.float32)
        postings_docs: Dict[str, List[int]] = {}
        postings_tfs: Dict[str, List[int]] = {}

        channels = posts_df["channel"].fillna("").astype(str).tolist()
        texts = posts_df["text"].fillna("").astype(str).tolist()

        for i in range(n_docs):
            full_text = f"{channels[i]} {texts[i]}"
            tokens = tokenize_stems(full_text)
            doc_len[i] = len(tokens)
            counts = Counter(tokens)
            for term, tf in counts.items():
                if term not in postings_docs:
                    postings_docs[term] = []
                    postings_tfs[term] = []
                postings_docs[term].append(i)
                postings_tfs[term].append(tf)

        avgdl = float(doc_len.mean()) if n_docs > 0 else 1.0
        self.doc_norm = np.asarray(
            self.cfg.bm25_k1 * (1.0 - self.cfg.bm25_b + self.cfg.bm25_b * doc_len / max(avgdl, 1.0)),
            dtype=np.float32,
        )

        self.inv_index.clear()
        self.idf.clear()
        for term, d_list in postings_docs.items():
            df = len(d_list)
            # Отсекаем сверхчастые шумовые токены (> 25% корпуса) на полноразмерном архиве
            if n_docs > 500 and df > n_docs * 0.25 and not term.isdigit():
                continue
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.inv_index[term] = (
                np.asarray(d_list, dtype=np.int32),
                np.asarray(postings_tfs[term], dtype=np.float32),
            )
        print(f"[BM25] Проиндексировано {n_docs} постов ({len(self.inv_index)} термов) за {time.time() - t0:.1f} с")

    def _prepare_chunks(self, posts_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        chunk_texts: List[str] = []
        chunk_parents: List[int] = []

        channels = posts_df["channel"].fillna("").astype(str).tolist()
        dates = posts_df["date"].fillna("").astype(str).tolist()
        texts = posts_df["text"].fillna("").astype(str).tolist()

        for doc_idx, (ch, dt, txt) in enumerate(zip(channels, dates, texts)):
            clean_txt = " ".join(txt.split())
            words = clean_txt.split(" ")
            header = f"passage: [{dt} | {ch}] "
            if len(words) <= self.cfg.chunk_words:
                chunk_texts.append(header + clean_txt)
                chunk_parents.append(doc_idx)
            else:
                n_chunks = 0
                for start in range(0, len(words), self.cfg.chunk_stride):
                    segment = " ".join(words[start : start + self.cfg.chunk_words])
                    if not segment:
                        break
                    chunk_texts.append(header + segment)
                    chunk_parents.append(doc_idx)
                    n_chunks += 1
                    if n_chunks >= self.cfg.max_chunks_per_post or start + self.cfg.chunk_words >= len(words):
                        break

        return chunk_texts, np.asarray(chunk_parents, dtype=np.int32)

    def build_dense_index(self, posts_df: pd.DataFrame) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[Dense] Загрузка энкодера {self.cfg.encoder_id} на {self.device}...")
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.encoder_id, revision=self.cfg.encoder_revision
        )
        self.encoder_model = AutoModel.from_pretrained(
            self.cfg.encoder_id, revision=self.cfg.encoder_revision, dtype=dtype
        ).to(self.device).eval()

        cache_path = (
            self.cfg.cache_dir / "dense_chunks_index.npz"
            if self.cfg.cache_dir is not None
            else None
        )
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path)
            if int(data["n_docs"]) == len(posts_df):
                self.chunk_embs = data["chunk_embs"]
                self.chunk_to_doc = data["chunk_to_doc"]
                self.doc_primary_emb = data["doc_primary_emb"]
                print(f"[Dense] Загружен кэш эмбеддингов из {cache_path}")
                return

        t0 = time.time()
        chunk_texts, self.chunk_to_doc = self._prepare_chunks(posts_df)
        print(f"[Dense] Сформировано {len(chunk_texts)} чанков для {len(posts_df)} постов.")

        self.chunk_embs = self._encode_texts(chunk_texts)

        # Сохраняем первый чанк каждого документа для быстрого поиска кросс-канальных репостов
        first_occurrence = np.zeros(len(posts_df), dtype=np.int32)
        seen = set()
        for chunk_idx, doc_idx in enumerate(self.chunk_to_doc):
            d_i = int(doc_idx)
            if d_i not in seen:
                seen.add(d_i)
                first_occurrence[d_i] = chunk_idx
        self.doc_primary_emb = self.chunk_embs[first_occurrence]

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                n_docs=len(posts_df),
                chunk_embs=self.chunk_embs,
                chunk_to_doc=self.chunk_to_doc,
                doc_primary_emb=self.doc_primary_emb,
            )
        print(f"[Dense] Индексация завершена за {time.time() - t0:.1f} с")

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if self.encoder_model is None or self.encoder_tokenizer is None:
            raise RuntimeError("Энкодер не инициализирован.")

        order = np.argsort([len(t) for t in texts])
        sorted_texts = [texts[i] for i in order]
        batches: List[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(sorted_texts), self.cfg.enc_batch_size):
                batch = sorted_texts[start : start + self.cfg.enc_batch_size]
                tok = self.encoder_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.enc_max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.encoder_model(**tok).last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                normed = F.normalize(pooled.float(), p=2, dim=1).cpu().numpy().astype(np.float16)
                batches.append(normed)

        embs = np.empty((len(texts), batches[0].shape[1]), dtype=np.float16)
        embs[order] = np.vstack(batches)
        return embs

    def index_corpus(self, posts_df: pd.DataFrame, use_dense: bool = True) -> None:
        self.posts_df = posts_df.reset_index(drop=True)
        self.post_ids = self.posts_df["post_id"].to_numpy(dtype=np.int64)
        self.post_dates = self.posts_df["date"].to_numpy(dtype="datetime64[D]")
        self.post_channels = self.posts_df["channel"].fillna("").astype(str).tolist()

        self.build_bm25(self.posts_df)
        if use_dense:
            self.build_dense_index(self.posts_df)

    def _bm25_scores(self, query: str, allowed_mask: np.ndarray) -> np.ndarray:
        assert self.doc_norm is not None
        scores = np.zeros(len(self.doc_norm), dtype=np.float32)
        k1 = self.cfg.bm25_k1
        for term in set(tokenize_stems(query)):
            if term not in self.inv_index:
                continue
            docs, tfs = self.inv_index[term]
            scores[docs] += self.idf[term] * (tfs * (k1 + 1.0)) / (tfs + self.doc_norm[docs])
        scores[~allowed_mask] = 0.0
        return scores

    def _dense_scores(self, query: str, allowed_mask: np.ndarray) -> np.ndarray:
        if self.chunk_embs is None or self.chunk_to_doc is None:
            return np.zeros(len(allowed_mask), dtype=np.float32)
        q_emb = self._encode_texts([f"query: {query}"])[0].astype(np.float32)
        chunk_sims = self.chunk_embs.astype(np.float32) @ q_emb

        doc_scores = np.full(len(allowed_mask), -1.0, dtype=np.float32)
        np.maximum.at(doc_scores, self.chunk_to_doc, chunk_sims)
        doc_scores[~allowed_mask] = -1.0
        return doc_scores

    def search_subqueries(
        self,
        subqueries: Sequence[str],
        asked_at: str,
        pool_size: int = 40,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Выполняет поиск по всем подзапросам вопроса строго на множестве постов с date <= asked_at,
        объединяет ранги через Reciprocal Rank Fusion (RRF) и возвращает индексы лучших постов
        вместе с диагностическими скорами (для детекции no_info).
        """
        assert self.post_dates is not None
        cutoff = np.datetime64(asked_at, "D")
        allowed = self.post_dates <= cutoff
        if not np.any(allowed):
            return np.array([], dtype=np.int64), {"max_lex": 0.0, "max_sem": 0.0}

        rrf_scores: Dict[int, float] = {}
        max_lex_overall = 0.0
        max_sem_overall = -1.0

        for q_idx, sq in enumerate(subqueries):
            weight = 1.0 if q_idx == 0 else 0.65
            lex = self._bm25_scores(sq, allowed)
            if len(lex) > 0:
                max_lex_overall = max(max_lex_overall, float(lex.max()))

            k_lex = min(self.cfg.lex_pool, int((lex > 0).sum()))
            if k_lex > 0:
                top_lex = np.argpartition(lex, -k_lex)[-k_lex:]
                top_lex = top_lex[np.argsort(-lex[top_lex])]
                for r, doc_i in enumerate(top_lex):
                    rrf_scores[int(doc_i)] = rrf_scores.get(int(doc_i), 0.0) + weight / (self.cfg.rrf_k + r + 1)

            if self.chunk_embs is not None:
                sem = self._dense_scores(sq, allowed)
                max_sem_overall = max(max_sem_overall, float(sem.max()))
                k_sem = min(self.cfg.sem_pool, int((sem > -1.0).sum()))
                if k_sem > 0:
                    top_sem = np.argpartition(sem, -k_sem)[-k_sem:]
                    top_sem = top_sem[np.argsort(-sem[top_sem])]
                    for r, doc_i in enumerate(top_sem):
                        rrf_scores[int(doc_i)] = rrf_scores.get(int(doc_i), 0.0) + weight / (self.cfg.rrf_k + r + 1)

        ranked_docs = sorted(rrf_scores.keys(), key=lambda i: (-rrf_scores[i], int(self.post_ids[i])))
        return (
            np.asarray(ranked_docs[:pool_size], dtype=np.int64),
            {"max_lex": max_lex_overall, "max_sem": max_sem_overall},
        )


# =============================================================================
# 4. ХРОНОЛОГИЧЕСКИЙ ОТБОР КОНТЕКСТА И ОПТИМИЗАЦИЯ AP@5
# =============================================================================

TIMELINE_SHIFT_RE = re.compile(
    r"\b(перенес\w+|отложил\w+|продлил\w+|новый\s+срок|вместо|сдвинул\w+|открыл\w+|запустил\w+|завершил\w+|отменил\w+)\b",
    re.I,
)


def select_chronological_context(
    candidate_indices: np.ndarray,
    posts_df: pd.DataFrame,
    question: str,
    top_k: int = 10,
) -> List[int]:
    """
    Формирует сбалансированный набор постов для LLM:
    1) Берёт топ-(top_k - 3) самых релевантных постов по RRF.
    2) Обязательно добавляет самый ранний и самый поздний релевантный пост из топ-25
       (чтобы не потерять первое обещание для C5/C6 и финальное состояние перед D для C1/B1).
    3) Для вопросов про переносы/сроки добирает посты с маркерами изменения дат.
    4) Возвращает выбранные индексы, отсортированные в ХРОНОЛОГИЧЕСКОМ порядке (date ASC).
    """
    if len(candidate_indices) == 0:
        return []
    if len(candidate_indices) <= top_k:
        selected = list(map(int, candidate_indices))
    else:
        selected = list(map(int, candidate_indices[: max(1, top_k - 3)]))
        top_window = list(map(int, candidate_indices[: min(25, len(candidate_indices))]))

        # Самый ранний и самый свежий пост среди уверенных кандидатов
        by_date = sorted(top_window, key=lambda idx: str(posts_df.iloc[idx]["date"]))
        earliest, latest = by_date[0], by_date[-1]
        for special_idx in (earliest, latest):
            if special_idx not in selected and len(selected) < top_k:
                selected.append(special_idx)

        # Посты с маркерами переносов/открытия/отмены
        for idx in top_window:
            if len(selected) >= top_k:
                break
            if idx not in selected and TIMELINE_SHIFT_RE.search(str(posts_df.iloc[idx]["text"])):
                selected.append(idx)

        # Добираем по RRF до top_k
        for idx in candidate_indices:
            if len(selected) >= top_k:
                break
            if int(idx) not in selected:
                selected.append(int(idx))

    # Сортируем по дате (от старых к новым), чтобы модель видела эволюцию истории
    selected_chrono = sorted(
        selected,
        key=lambda idx: (str(posts_df.iloc[idx]["date"]), int(posts_df.iloc[idx]["post_id"])),
    )
    return selected_chrono


def expand_evidence_for_ap5(
    primary_post_ids: List[int],
    candidate_indices: np.ndarray,
    retriever: TemporalHybridRetriever,
    max_k: int = 5,
) -> List[int]:
    """
    Оптимизирует колонку `evidence` под метрику eq = AP@5:
    В архиве одну и ту же новость часто публикуют сразу несколько городских пабликов
    (в тот же день или на следующий). Все такие пересказы входят в золотое множество G_q.
    Эта функция сохраняет выбранные LLM опорные посты первыми и добирает до 5 постов
    за счёт ближайших семантических/лексических дубликатов среди кандидатов <= D.
    """
    assert retriever.posts_df is not None and retriever.post_ids is not None
    pid_to_idx = {int(retriever.post_ids[i]): int(i) for i in candidate_indices}

    ordered_pids: List[int] = []
    for pid in primary_post_ids:
        if pid in pid_to_idx and pid not in ordered_pids:
            ordered_pids.append(pid)

    # Если LLM не указала валидные post_id, берём лучший пост из RRF
    if not ordered_pids and len(candidate_indices) > 0:
        ordered_pids.append(int(retriever.post_ids[candidate_indices[0]]))

    # Ищем кросс-канальные пересказы для первого опорного поста среди пула кандидатов
    if ordered_pids and retriever.doc_primary_emb is not None and len(ordered_pids) < max_k:
        anchor_idx = pid_to_idx[ordered_pids[0]]
        anchor_emb = retriever.doc_primary_emb[anchor_idx].astype(np.float32)
        anchor_tokens = set(tokenize_stems(str(retriever.posts_df.iloc[anchor_idx]["text"])))

        scored_dups: List[Tuple[float, int]] = []
        for cand_idx in candidate_indices[:30]:
            c_i = int(cand_idx)
            c_pid = int(retriever.post_ids[c_i])
            if c_pid in ordered_pids:
                continue
            cand_emb = retriever.doc_primary_emb[c_i].astype(np.float32)
            cos_sim = float(np.dot(anchor_emb, cand_emb))

            cand_tokens = set(tokenize_stems(str(retriever.posts_df.iloc[c_i]["text"])))
            jacc = (
                len(anchor_tokens & cand_tokens) / max(1, len(anchor_tokens | cand_tokens))
                if anchor_tokens
                else 0.0
            )
            combined = 0.75 * cos_sim + 0.25 * jacc
            if cos_sim >= retriever.cfg.cross_channel_dup_sim or jacc >= 0.40:
                scored_dups.append((combined, c_pid))

        scored_dups.sort(reverse=True)
        for _, dup_pid in scored_dups:
            if len(ordered_pids) >= max_k:
                break
            ordered_pids.append(dup_pid)

    # Добираем оставшиеся слоты до 5 лучшими кандидатами из RRF
    for cand_idx in candidate_indices:
        if len(ordered_pids) >= max_k:
            break
        c_pid = int(retriever.post_ids[int(cand_idx)])
        if c_pid not in ordered_pids:
            ordered_pids.append(c_pid)

    return ordered_pids[:max_k]


# =============================================================================
# 5. ПРОМПТИНГ ЧИТАТЕЛЯ (LLM READER) И ПОСТОБРАБОТКА ОТВЕТОВ
# =============================================================================

SYSTEM_PROMPT = """Ты — аналитик архива городских сообществ Нижегородской области.
Вопрос задан в дату D. Посты после D не существуют. Отвечай СТРОГО по приведённым постам (они отсортированы хронологически от ранних к более свежим).

ПРАВИЛА ВЫВОДА:
1. Если на дату D событие было только запланировано или обещано, но в постах до D включительно НЕТ подтверждения, что оно уже состоялось:
   - при формате status верни код статуса: promised (или postponed, если срок перенесли на новую дату; или cancelled, если отменили);
   - при любом другом формате (date, number, yesno, choice), если вопрос спрашивает о свершившемся факте (например, «С какого числа открыли...», «Сколько стоил... после запуска»), но к дате D есть только обещание без факта выполнения, верни promised.
2. Если в постах до D вообще нет сведений для ответа на вопрос (например, деталь не упоминалась в сводке или событие ещё не анонсировали), верни no_info.
3. Если изменились тарифы, сроки или показатели, бери самое актуальное значение на дату D (если вопрос не спрашивает про первоначальный план или разницу).
4. Для вопросов про разницу в днях или полных месяцах между первым обещанием и фактическим событием найди самый ранний пост с обещанием и пост с фактом выполнения и точно посчитай разницу.
5. Внимательно проверяй варианты choice (a, b, c, d): если сам вопрос содержит ложную предпосылку, а один из вариантов прямо указывает на это («аварии не было...», «в 2023 году не запускали...»), выбирай этот опровергающий вариант.

Формат ответа в поле ANSWER:
- date: ГГГГ-ММ-ДД
- number: только число (разделитель дробной части — точка, без единиц измерения и пробелов), в единицах поля UNIT
- status: ровно одно слово из done, promised, postponed, cancelled
- yesno: yes или no
- choice: ровно одна буква из a, b, c, d
- специальные значения (для любого формата): promised, no_info

Выведи ровно 3 строки:
REASONING: краткий расчёт дат/чисел и проверка факта выполнения к дате D (1–2 предложения)
ANSWER: <ответ строго по формату>
EVIDENCE: <до 5 post_id через пробел, из которых прямо следует ответ; пусто для no_info>"""


FORMAT_HINTS = {
    "date": "ГГГГ-ММ-ДД (или promised / no_info)",
    "number": "число через точку без единиц измерения (или promised / no_info)",
    "status": "done | promised | postponed | cancelled | no_info",
    "yesno": "yes | no | promised | no_info",
    "choice": "a | b | c | d | promised | no_info",
}


def build_reader_prompt(row: pd.Series, chrono_indices: Sequence[int], posts_df: pd.DataFrame) -> str:
    asked_dt = datetime.strptime(str(row["asked_at"]), "%Y-%m-%d").date()
    blocks: List[str] = []

    for idx in chrono_indices:
        p = posts_df.iloc[idx]
        p_date_str = str(p["date"])
        try:
            p_dt = datetime.strptime(p_date_str, "%Y-%m-%d").date()
            delta_days = (asked_dt - p_dt).days
            delta_label = f"за {delta_days} дн. до D" if delta_days > 0 else "в день вопроса D"
        except ValueError:
            delta_label = p_date_str

        ch = str(p.get("channel", "")).strip()
        txt = " ".join(str(p.get("text", "")).split())
        blocks.append(f"[post_id={p['post_id']} | дата={p_date_str} ({delta_label}) | канал={ch}]\n{txt}")

    meta = [
        f"Дата вопроса D: {row['asked_at']}",
        f"Вопрос: {row['question']}",
        f"Требуемый формат (answer_format): {row['answer_format']} ({FORMAT_HINTS.get(str(row['answer_format']), '')})",
    ]
    if pd.notna(row.get("unit")) and str(row.get("unit")).strip():
        meta.append(f"Единица измерения (UNIT): {row['unit']}")
    if pd.notna(row.get("options")) and str(row.get("options")).strip():
        meta.append(f"Варианты ответа:\n{row['options']}")

    posts_section = "\n\n".join(blocks) if blocks else "(Подходящих постов до даты D не найдено)"
    return "\n".join(meta) + "\n\nХронология постов архива (<= D):\n" + posts_section


# Регулярные выражения валидации форматов (строго по submission.py)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
VALID_FIXED = {
    "status": {"done", "promised", "postponed", "cancelled"},
    "yesno": {"yes", "no"},
    "choice": {"a", "b", "c", "d"},
}
SPECIAL_ANSWERS = {"promised", "no_info", "skip"}


def normalize_raw_answer(raw: str, fmt: str, unit: Optional[str] = None) -> str:
    """Приводит сырой ответ LLM к строгому формату, требуемому `submission.py`."""
    s = (raw or "").strip()
    if not s:
        return "skip"

    low = s.lower().strip(" .,;:!?'\"«»()")
    if low in SPECIAL_ANSWERS:
        return low
    for sp in ("no_info", "promised", "skip"):
        if re.search(rf"\b{sp}\b", low):
            return sp

    if fmt in VALID_FIXED:
        if low in VALID_FIXED[fmt]:
            return low
        if fmt == "choice":
            m = re.match(r"^([a-d])(?:\)|\.|\b)", low)
            if m:
                return m.group(1)
        if fmt == "yesno":
            if low in ("да", "верно", "true"):
                return "yes"
            if low in ("нет", "неверно", "false"):
                return "no"
        if fmt == "status":
            for code in ("done", "promised", "postponed", "cancelled"):
                if code in low:
                    return code
        return "skip"

    if fmt == "date":
        m_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", s)
        if m_iso:
            cand = m_iso.group(1)
            try:
                date.fromisoformat(cand)
                return cand
            except ValueError:
                pass
        m_ru = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", s)
        if m_ru:
            cand = f"{m_ru.group(3)}-{m_ru.group(2)}-{m_ru.group(1)}"
            try:
                date.fromisoformat(cand)
                return cand
            except ValueError:
                pass
        return "skip"

    if fmt == "number":
        # Заменяем запятую на точку и убираем пробелы-разделители тысяч (например, "12 500" -> "12500")
        s_clean = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", s.replace(",", "."))
        m_num = re.search(r"-?\d+(?:\.\d+)?", s_clean)
        if m_num:
            val_str = m_num.group(0)
            try:
                val = float(val_str)
                if math.isfinite(val):
                    if val.is_integer():
                        return str(int(val))
                    return f"{val:.4f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
        return "skip"

    return "skip"


def parse_llm_output(
    raw_output: str,
    fmt: str,
    unit: Optional[str],
    allowed_post_ids: set[int],
) -> Tuple[str, List[int]]:
    """Извлекает ANSWER и EVIDENCE из ответа модели и валидирует их."""
    text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.S).strip()

    ans_match = re.search(r"^ANSWER\s*:\s*(.+)$", text, flags=re.M | re.I)
    ev_match = re.search(r"^EVIDENCE\s*:\s*(.*)$", text, flags=re.M | re.I)

    raw_ans = ans_match.group(1).strip() if ans_match else (text.splitlines()[-1] if text else "")
    ans = normalize_raw_answer(raw_ans, fmt, unit)

    ev_ids: List[int] = []
    if ev_match:
        for tok in re.findall(r"\d+", ev_match.group(1)):
            pid = int(tok)
            if pid in allowed_post_ids and pid not in ev_ids:
                ev_ids.append(pid)

    return ans, ev_ids[:5]


# =============================================================================
# 6. ОСНОВНОЙ КОНВЕЙЕР РЕШЕНИЯ (ОБУЧЕНИЕ АЛИАСОВ + ИНФЕРЕНС + ПРОВЕРКА)
# =============================================================================

class ArchiveQAPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.expander = NizhnyNovgorodQueryExpander()
        self.retriever = TemporalHybridRetriever(cfg)
        self.llm_tokenizer = None
        self.llm_model = None

    def load_llm(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"[LLM] Загрузка читателя {self.cfg.llm_id} на {device}...")

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.llm_id, revision=self.cfg.llm_revision
        )
        # Поддержка как Qwen3_5ForConditionalGeneration, так и стандартного AutoModelForCausalLM
        try:
            from transformers import Qwen3_5ForConditionalGeneration
            self.llm_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.cfg.llm_id,
                revision=self.cfg.llm_revision,
                dtype=dtype,
                attn_implementation="sdpa",
            ).to(device).eval()
        except Exception:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.llm_id,
                revision=self.cfg.llm_revision,
                torch_dtype=dtype,
            ).to(device).eval()

    def generate_answer(self, user_prompt: str) -> str:
        import torch

        assert self.llm_model is not None and self.llm_tokenizer is not None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            chat_text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.llm_tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_tokens,
        ).to(self.llm_model.device)

        with torch.inference_mode():
            out_ids = self.llm_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
        return self.llm_tokenizer.decode(gen_ids, skip_special_tokens=True)

    def predict_questions(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        assert self.retriever.posts_df is not None and self.retriever.post_ids is not None
        if self.llm_model is None:
            self.load_llm()

        rows_out: List[Dict[str, str]] = []
        t0 = time.time()

        for idx, (_, row) in enumerate(questions_df.iterrows(), 1):
            qid = str(row["qid"])
            question = str(row["question"])
            asked_at = str(row["asked_at"])
            fmt = str(row["answer_format"])
            unit = str(row["unit"]) if pd.notna(row.get("unit")) else None
            options = str(row["options"]) if pd.notna(row.get("options")) else None

            subqueries = self.expander.expand_query(question, options)
            cand_indices, diag = self.retriever.search_subqueries(
                subqueries, asked_at=asked_at, pool_size=40
            )

            # Быстрый детектор отсутствия информации в архиве до даты D
            if (
                len(cand_indices) == 0
                or (
                    diag["max_lex"] < self.cfg.no_info_max_lex_score
                    and diag["max_sem"] < self.cfg.no_info_max_sem_score
                )
            ):
                rows_out.append({"qid": qid, "answer": "no_info", "evidence": ""})
                continue

            chrono_indices = select_chronological_context(
                cand_indices,
                self.retriever.posts_df,
                question=question,
                top_k=self.cfg.reader_top_k,
            )
            allowed_pids = {int(self.retriever.post_ids[i]) for i in cand_indices}

            prompt = build_reader_prompt(row, chrono_indices, self.retriever.posts_df)
            raw_reply = self.generate_answer(prompt)
            ans, llm_ev_pids = parse_llm_output(raw_reply, fmt=fmt, unit=unit, allowed_post_ids=allowed_pids)

            # Оформление доказательств с учётом правил метрики
            if ans in ("no_info", "skip"):
                ev_str = ""
            else:
                expanded_pids = expand_evidence_for_ap5(
                    llm_ev_pids, cand_indices, self.retriever, max_k=self.cfg.evidence_max_k
                )
                ev_str = " ".join(str(pid) for pid in expanded_pids)

            rows_out.append({"qid": qid, "answer": ans, "evidence": ev_str})

            if idx % 20 == 0 or idx == len(questions_df):
                elapsed = time.time() - t0
                speed = elapsed / idx
                eta = speed * (len(questions_df) - idx)
                print(f"[{idx}/{len(questions_df)}] Затрачено: {elapsed:.1f} с | Осталось: ~{eta:.1f} с")

        return pd.DataFrame(rows_out, columns=["qid", "answer", "evidence"])


# =============================================================================
# 7. CLI И АВТОМАТИЧЕСКАЯ ПРОВЕРКА ФОРМАТА (submission.py / score.py)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Решение Кейса 2B «Спроси архив» (НТО ИИ 2026/27)")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Папка с corpus.parquet, train.csv, test.csv")
    parser.add_argument("--mode", choices=["test", "val", "smoke"], default="test", help="Режим запуска")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Путь для сохранения ответа")
    parser.add_argument("--val-limit", type=int, default=100, help="Число вопросов для быстрой локальной валидации")
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        output_csv=args.output,
        cache_dir=args.data_dir / ".cache_2b",
    )

    if args.mode == "smoke":
        # Быстрая самопроверка всех компонентов (экспандер, BM25, хронология, парсер ответов) без тяжёлых весов
        print("[Smoke] Запуск самопроверки модулей на синтетическом мини-корпусе...")
        dummy_posts = pd.DataFrame([
            {"post_id": 10, "date": "2024-06-01", "channel": "Мой Нижний", "channel_type": "vk",
             "text": "Крыши над спусками в метро на станции Двигатель Революции построят за 40.5 млн рублей к осени."},
            {"post_id": 11, "date": "2024-06-02", "channel": "Типичный НН", "channel_type": "tg",
             "text": "Павильоны над сходами в подземку у Двигателя Революции обойдутся бюджету в 40,5 млн руб."},
            {"post_id": 25, "date": "2024-07-20", "channel": "Мой Нижний", "channel_type": "vk",
             "text": "Стоимость ремонта крыш над спусками метро Двигатель Революции выросла до 42.7 млн руб., срок перенесли."},
        ])
        expander = NizhnyNovgorodQueryExpander()
        q = "Во сколько обходятся крыши над спусками в подземку у станции между Заречной и Пролетарской?"
        subs = expander.expand_query(q)
        assert any("Двигатель Революции" in s for s in subs), "Перифраза станции не раскрылась!"

        retr = TemporalHybridRetriever(cfg)
        retr.index_corpus(dummy_posts, use_dense=False)

        cands_july, _ = retr.search_subqueries(subs, asked_at="2024-07-01")
        assert 25 not in [int(retr.post_ids[i]) for i in cands_july], "Нарушен темпоральный фильтр <= D!"
        ev_july = expand_evidence_for_ap5([10], cands_july, retr, max_k=5)
        assert ev_july[:2] == [10, 11], f"Ожидались посты [10, 11], получено {ev_july}"

        assert normalize_raw_answer("42,70 млн руб.", "number", "млн руб") == "42.7"
        assert normalize_raw_answer("30.11.2024", "date") == "2024-11-30"
        assert normalize_raw_answer("b) Канатная дорога", "choice") == "b"
        print("[Smoke] Все проверки успешно пройдены!")
        return

    corpus_path = cfg.data_dir / "corpus.parquet"
    train_path = cfg.data_dir / "train.csv"
    test_path = cfg.data_dir / "test.csv"

    print(f"[Init] Чтение корпуса из {corpus_path}...")
    posts_df = pd.read_parquet(corpus_path)
    train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()

    pipeline = ArchiveQAPipeline(cfg)
    if not train_df.empty:
        pipeline.expander.fit_from_train(train_df, posts_df)

    pipeline.retriever.index_corpus(posts_df, use_dense=True)

    if args.mode == "val":
        if train_df.empty:
            raise FileNotFoundError(f"Не найден {train_path} для режима val.")
        val_df = train_df.head(args.val_limit).copy()
        pred_df = pipeline.predict_questions(val_df)
        pred_df.to_csv(cfg.output_csv, index=False)
        print(f"[Val] Предсказания сохранены в {cfg.output_csv}")

        try:
            from score import evaluate
            res = evaluate(val_df, pred_df, posts_df)
            print(f"[Val] Итоговый балл S = {res['score']:.2f} | По уровням: {res['by_level']}")
        except ImportError:
            print("[Val] Модуль score.py не найден в текущей директории.")
    else:
        questions_df = pd.read_csv(test_path)
        pred_df = pipeline.predict_questions(questions_df)
        pred_df.to_csv(cfg.output_csv, index=False)
        print(f"[Test] Предсказания сохранены в {cfg.output_csv}")

        try:
            from submission import check
            errors = check(questions_df, pred_df, set(posts_df["post_id"].astype(int)))
            if errors:
                print(f"[Check] ВНИМАНИЕ! Найдены ошибки формата ({len(errors)}): {errors[:5]}")
            else:
                print(f"[Check] OK: {len(pred_df)} строк, все проверки submission.py пройдены!")
        except ImportError:
            print("[Check] Модуль submission.py не найден в текущей директории.")


if __name__ == "__main__":
    main()