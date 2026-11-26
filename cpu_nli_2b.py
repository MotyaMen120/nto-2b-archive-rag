"""Optional local Russian entailment features for a categorical reader."""

from functools import lru_cache
from pathlib import Path
import json
import re

import numpy as np


def statement(question):
    claim = re.sub(r"^\s*(?:ну что,?\s*|правда(?: ли)?,?\s*(?:что\s*)?|верно ли,?\s*(?:что\s*)?)",
                   "", question, flags=re.I).strip(" ?.")
    claim = re.sub(r"\s+ли\b", "", claim, flags=re.I)
    claim = re.split(r",?\s+или пока\b", claim, flags=re.I)[0].strip()
    return claim[:1].upper() + claim[1:] + "."


def choice_claim(question, option):
    if "предпосылка неверна" in option:
        return re.sub(r"^.*?предпосылка неверна[: ,–—-]*", "", option)
    # In wh-questions the option often supplies a location, team, date, etc.
    segments = re.split(r"\b(?:в каком районе|какому сроку|какому году|какому месяцу|какой команд[еыу]|какой район|какой месяц|каком году|какой год|какое место|каком месте|сколько человек|сколько|где|куда|когда|откуда|почему|с каким сч[её]том|чем)\b",
                        question.strip(" ?"), maxsplit=1, flags=re.I)
    if len(segments) == 2:
        return (segments[0] + " " + option + " " + segments[1]).strip() + "."
    return f"{question} {option}."


@lru_cache(maxsize=1)
def _model(path):
    from tokenizers import Tokenizer
    folder = Path(path)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=152, strategy="longest_first")
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    onnx = next((folder / n for n in ("model_int8.onnx", "model.onnx") if (folder / n).is_file()), None)
    if onnx:
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 3
        session_options.inter_op_num_threads = 1
        model = ort.InferenceSession(str(onnx), session_options, providers=["CPUExecutionProvider"])
    else:
        import torch
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(folder, local_files_only=True).eval()
        torch.set_num_threads(min(3, torch.get_num_threads()))
    config = json.loads((folder / "config.json").read_text(encoding="utf8"))
    labels = {str(k).lower(): int(v) for k, v in config["label2id"].items()}
    if "entailment" not in labels:
        raise ValueError("Entailment model config needs an entailment output label")
    return tokenizer, model, labels


def compute(path, q, fmt, options, visible):
    return _features(str(Path(path).resolve()), q, fmt, tuple(options), tuple(visible))


@lru_cache(maxsize=5000)
def _features(path, q, fmt, options, visible):
    from cpu_categorical_2b import clean, sentences, words
    claims = [choice_claim(q, opt) for opt in options] if fmt == "choice" else [statement(q)]
    if fmt == "status":
        claims += ["Событие только планируется.", "Срок перенесли на более позднюю дату.", "Событие отменили."]
    if not visible or not claims:
        return [{} for _ in claims]
    qt = words(clean(q))
    premises = []
    seen = set()
    for _, raw in visible:
        clauses = sentences(raw)
        if not clauses:
            continue
        # Retain an adjacent sentence, which can supply the date or the entity.
        index = max(range(len(clauses)), key=lambda i: len(words(clauses[i]) & qt) /
                    (4 + len(words(clauses[i]))) ** .3)
        neighbors = [clauses[index]]
        if index:
            neighbors.append(clauses[index-1])
        elif index + 1 < len(clauses):
            neighbors.append(clauses[index+1])
        text = " ".join(neighbors)[:1020]
        if text not in seen:
            premises.append(text)
            seen.add(text)
    if not premises:
        return [{} for _ in claims]
    tokenizer, model, labels = _model(path)
    batch = tokenizer.encode_batch([(p, claim) for claim in claims for p in premises])
    inputs = {name: np.array([getattr(b, attr) for b in batch], dtype=np.int64)
              for name, attr in (("input_ids", "ids"), ("token_type_ids", "type_ids"), ("attention_mask", "attention_mask"))}
    if hasattr(model, "get_inputs"):
        feed = {i.name: inputs[i.name] for i in model.get_inputs()}
        logits = model.run(None, feed)[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        logits = exp / exp.sum(axis=1, keepdims=True)
    else:
        import torch
        with torch.inference_mode():
            logits = model(**{k: torch.from_numpy(v) for k,v in inputs.items()}).logits.softmax(dim=1).cpu().numpy()
    logits = logits.reshape(len(claims), len(premises), -1)
    features = []
    for predictions in logits:
        feature = {}
        for label, index in labels.items():
            values = predictions[:, index]
            weights = 1 / np.sqrt(1 + np.arange(len(values)) / 2)
            feature["nli_top_" + label] = float(values[0])
            feature["nli_mean_" + label] = float(np.average(values, weights=weights))
            feature["nli_max_" + label] = float(np.max(values * weights))
        features.append(feature)
    return features
