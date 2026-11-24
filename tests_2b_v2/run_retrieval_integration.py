"""Check local Transformers encoder/CE APIs using tiny random models, no downloads."""
import gc
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import solution_2b_v2 as s


def main():
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        encoder, reranker = root / "encoder", root / "reranker"
        vocab = root / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "мост", "парк", "открыли", "анонс", "passage", "query"]), encoding="utf-8")
        tokenizer = BertTokenizer(vocab_file=str(vocab))
        config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_labels=1)
        BertModel(config).save_pretrained(encoder)
        BertForSequenceClassification(config).save_pretrained(reranker)
        tokenizer.save_pretrained(encoder)
        tokenizer.save_pretrained(reranker)
        cfg = s.Config(cache=root / "cache", encoder=str(encoder), reranker=str(reranker), batch=2)
        posts = pd.DataFrame([
            {"post_id": 1, "date": "2024-02-01", "channel": "A", "channel_type": "vk", "text": "Анонс открытия моста."},
            {"post_id": 2, "date": "2024-02-02", "channel": "B", "channel_type": "vk", "text": "Мост открыли."},
            {"post_id": 3, "date": "2025-02-01", "channel": "C", "channel_type": "tg", "text": "Мост снова открыли."},
        ])
        questions = pd.DataFrame([{"qid": "synthetic", "question": "Открыли мост?", "asked_at": "2024-02-02", "answer_format": "yesno"}])
        with patch("torch.cuda.is_available", return_value=False):
            retriever = s.Retriever(posts, cfg)
            items = s.prepare(questions, retriever)
            assert isinstance(retriever.embeddings, np.memmap)
            assert retriever.embeddings.dtype == np.float16
            assert retriever.encoder_model is None
            assert set(items[0]["candidates"]) == {0, 1}
            assert all(retriever.ids[i] != 3 for i in items[0]["candidates"])
            assert np.isfinite(items[0]["ce_top"])
        # Release Windows file mappings before TemporaryDirectory cleanup.
        retriever.embeddings = retriever.parents = retriever.first = None
        gc.collect()
        print(json.dumps({"local_encoder_api": "passed", "local_cross_encoder_api": "passed", "future_post_excluded": True, "random_models_only": True}))


if __name__ == "__main__":
    main()
