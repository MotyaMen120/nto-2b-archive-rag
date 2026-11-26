"""Grouped, nested OOF evidence ranking. Reader inputs never use gold-trained ranks."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

FEATURES = ("bm25", "dense", "rrf_rank", "ce_logit", "ce_scored", "days_before", "log_age",
            "number_overlap", "term_overlap", "words", "channel_public", "channel_media", "channel_official",
            "target_distance", "candidate_earliest", "candidate_latest", "reposts_2d",
            "format_number", "format_date", "format_status", "format_yesno", "format_choice")


def retrieval_features(row, item, retriever):
    from solution_2b_v2 import tokens
    ids = item["retrieval"]
    hi = retriever.cutoff(row["asked_at"])
    query = item["plan"]["canonical_query"]
    lex = retriever.lexical(query, hi)
    sem = retriever.semantic(query, hi)
    query_terms = set(tokens(query))
    query_nums = {t for t in query_terms if any(c.isdigit() for c in t)}
    term_sets = {i: set(tokens(retriever.texts[i])) for i in ids}
    num_sets = {i: {t for t in term_sets[i] if any(c.isdigit() for c in t)} for i in ids}
    dates = retriever.dates
    oldest = min((dates[i] for i in ids), default=None)
    newest = max((dates[i] for i in ids), default=None)
    target_dates = np.asarray(item["plan"]["target_dates"], dtype="datetime64[D]")
    result = []
    for rank, i in enumerate(ids):
        age = int((np.datetime64(row["asked_at"], "D") - dates[i]).astype(int))
        cluster = 1
        for j in ids:
            if i == j or retriever.channels[i] == retriever.channels[j] or abs(int((dates[i] - dates[j]).astype(int))) > 2:
                continue
            if num_sets[i] and num_sets[j] and num_sets[i] != num_sets[j]:
                continue
            jaccard = len(term_sets[i] & term_sets[j]) / max(1, len(term_sets[i] | term_sets[j]))
            cosine = 0
            if retriever.embeddings is not None:
                cosine = float(np.dot(retriever.embeddings[retriever.first[i]].astype(np.float32), retriever.embeddings[retriever.first[j]].astype(np.float32)))
            if jaccard >= 0.55 or (cosine >= 0.85 and jaccard >= 0.15):
                cluster += 1
        kind = str(retriever.posts.iloc[i]["channel_type"])
        distance = int(np.min(np.abs((target_dates - dates[i]).astype(int)))) if len(target_dates) else 3660
        result.append([float(lex[i]), float(sem[i]) if np.isfinite(sem[i]) else -1, rank + 1, -20, 0, age, np.log1p(age),
                       len(query_nums & num_sets[i]) / max(1, len(query_nums)),
                       len(query_terms & term_sets[i]) / max(1, len(query_terms)), len(retriever.texts[i].split()),
                       kind == "Паблик", kind == "СМИ", kind == "Госструктура", distance,
                       dates[i] == oldest, dates[i] == newest, cluster,
                       *[row["answer_format"] == fmt for fmt in ["number", "date", "status", "yesno", "choice"]]])
    item["meta_candidates"] = [int(retriever.ids[i]) for i in ids]
    item["meta_features"] = np.asarray(result, dtype=np.float32).reshape(-1, len(FEATURES))


def attach_cross_encoder(item):
    scores = item.get("ce_scores", {})
    for n, pid in enumerate(item["meta_candidates"]):
        if pid in scores:
            item["meta_features"][n, 3] = scores[pid]
            item["meta_features"][n, 4] = 1


def fit_ranker(records, indices, gold_by_id):
    from lightgbm import LGBMRanker
    matrices, labels, sizes, weights = [], [], [], []
    for i in indices:
        record = records[i]
        if not record["meta_candidates"]:
            continue
        gold = gold_by_id[record["qid"]]
        relevant = set(gold["_G"])
        matrix = np.asarray(record["meta_features"], np.float32)
        if matrix.shape != (len(record["meta_candidates"]), len(FEATURES)) or not np.isfinite(matrix).all():
            raise ValueError("Invalid meta-ranker feature schema.")
        matrices.append(matrix)
        labels.extend(int(pid in relevant) for pid in record["meta_candidates"])
        sizes.append(len(matrix))
        weights.extend([gold["_w"]] * len(matrix))
    if not matrices:
        raise ValueError("No candidates available to train the meta-ranker.")
    model = LGBMRanker(objective="lambdarank", metric="map", n_estimators=140, num_leaves=15,
                       learning_rate=0.04, min_child_samples=30, reg_lambda=5.0,
                       random_state=42, n_jobs=4, verbosity=-1)
    model.fit(np.vstack(matrices), np.asarray(labels), group=sizes, sample_weight=np.asarray(weights), feature_name=list(FEATURES))
    return model


def apply_ranker(records, indices, model, retriever):
    result = []
    for i in indices:
        record = dict(records[i])
        ids = record["meta_candidates"]
        matrix = np.asarray(record["meta_features"], np.float32)
        predictor = getattr(model, "booster_", model)
        scores = predictor.predict(matrix) if ids else []
        order = sorted(range(len(ids)), key=lambda j: (-scores[j], ids[j]))
        ranked_pids = [ids[j] for j in order]
        candidate_indices = [retriever.by_id[p] for p in ranked_pids]
        record["fallback_evidence"] = ranked_pids[:5]
        if record["answer"] == "no_info":
            record["evidence"] = []
        elif record["answer"] == "skip":
            record["evidence"] = ranked_pids[:5]
        else:
            record["evidence"] = retriever.evidence(record.get("cited", []), candidate_indices)
        record["meta_applied"] = True
        result.append(record)
    return result


def new_calibrator():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=160, max_depth=5, min_samples_leaf=12, random_state=42, n_jobs=1)


def fit_calibrator(records, questions, scorer, gold_by_id):
    from solution_2b_v2 import calibration_features
    rows = [scorer.score_item(gold_by_id[r["qid"]], r) for r in records]
    return new_calibrator().fit(calibration_features(records, questions), [r["s"] + 0.175 for r in rows], sample_weight=[r["w"] for r in rows])


def apply_utility(records, questions, model):
    from solution_2b_v2 import calibration_features
    result = [dict(r) for r in records]
    for record, gain in zip(result, model.predict(calibration_features(records, questions))):
        record["expected_advantage"] = float(gain)
        if gain < 0:
            record.update(answer="skip", evidence=record["fallback_evidence"])
    return result


def train_nested_oof(records, questions, groups_path, splits, scorer, gold, retriever, output, signature, calibrate):
    import joblib
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    gold_by_id = {g["id"]: g for g in gold}
    group_map = pd.read_csv(groups_path, dtype=str).set_index("qid").group
    groups = questions.qid.map(group_map).to_numpy()
    raw_meta_oof = [None] * len(records)
    final_oof = [None] * len(records)
    for fold, (tr, va) in enumerate(splits):
        assert not set(groups[tr]) & set(groups[va])
        model = fit_ranker(records, tr, gold_by_id)
        ranked_val = apply_ranker(records, va, model, retriever)
        for idx, record in zip(va, ranked_val):
            raw_meta_oof[idx] = record
        if calibrate:
            # Each inner evidence list is built without any outer-validation gold.
            inner_records = [None] * len(tr)
            inner_groups = groups[tr]
            for inner_tr, inner_va in GroupKFold(n_splits=5).split(tr, groups=inner_groups):
                actual_train, actual_val = tr[inner_tr], tr[inner_va]
                assert not set(groups[actual_train]) & set(groups[va])
                assert not set(groups[actual_train]) & set(groups[actual_val])
                inner_model = fit_ranker(records, actual_train, gold_by_id)
                for pos, record in zip(inner_va, apply_ranker(records, actual_val, inner_model, retriever)):
                    inner_records[pos] = record
            utility = fit_calibrator(inner_records, questions.iloc[tr], scorer, gold_by_id)
            ranked_val = apply_utility(ranked_val, questions.iloc[va], utility)
        for idx, record in zip(va, ranked_val):
            final_oof[idx] = record
        print(f"Nested meta OOF fold {fold + 1}/5 complete", flush=True)
    final_ranker = fit_ranker(records, np.arange(len(records)), gold_by_id)
    joblib.dump({"signature": signature, "features": FEATURES, "model": final_ranker}, output.with_suffix(".meta.joblib"))
    if calibrate:
        utility = fit_calibrator(raw_meta_oof, questions, scorer, gold_by_id)
        joblib.dump({"signature": signature, "version": retriever_version(), "requires_meta": True, "model": utility}, output.with_suffix(".calibrator.joblib"))
    return final_oof


def retriever_version():
    from solution_2b_v2 import VERSION
    return VERSION


def load_apply(records, retriever, path, signature):
    import joblib
    artifact = joblib.load(path)
    if artifact.get("signature") != signature or tuple(artifact.get("features", ())) != FEATURES:
        raise ValueError("Meta-ranker was trained with a different pipeline/feature schema.")
    return apply_ranker(records, range(len(records)), artifact["model"], retriever)
