import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import solution_2b_v2 as s
import meta_2b as meta


class MetaAndResume(unittest.TestCase):
    def test_chunked_attention_matches_causal_padding_mask(self):
        import torch
        from types import SimpleNamespace
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        torch.manual_seed(7)
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        query = torch.randn(2, 4, 67, 16)
        key, value = torch.randn(2, 2, 67, 16), torch.randn(2, 2, 67, 16)
        positions = torch.arange(67)
        allowed = (positions[:, None] >= positions[None, :])[None, None].repeat(2, 1, 1, 1)
        allowed[1, :, :, :5] = False
        expected, _ = sdpa_attention_forward(module, query, key, value, allowed, scaling=0.25)
        actual, _ = s.attention_in_chunks(sdpa_attention_forward, module, query, key, value, allowed, scaling=0.25, block_size=16)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        expected, _ = sdpa_attention_forward(module, query, key, value, None, scaling=0.25)
        actual, _ = s.attention_in_chunks(sdpa_attention_forward, module, query, key, value, None, scaling=0.25, block_size=16)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_nested_folds_keep_outer_validation_out_of_inner_training(self):
        with tempfile.TemporaryDirectory() as folder:
            questions = pd.DataFrame([{"qid": f"q{i}", "question": f"Вопрос {i}", "evidence": str(1000 + i),
                                       "answer_format": "yesno", "level": "light"} for i in range(50)])
            groups_path = Path(folder) / "groups.csv"
            pd.DataFrame({"qid": questions.qid, "group": [str(i // 5) for i in range(50)]}).to_csv(groups_path, index=False)
            _, splits = s.group_folds(questions, groups_path)
            records = [{"qid": qid} for qid in questions.qid]
            fits = []
            def fit(records, indices, gold):
                training = set(map(int, indices))
                fits.append(training)
                return training
            def apply(records, indices, model, retriever):
                self.assertFalse(set(map(int, indices)) & model)
                return [{**records[i], "meta_applied": True} for i in indices]
            with patch.object(meta, "fit_ranker", side_effect=fit), patch.object(meta, "apply_ranker", side_effect=apply), \
                 patch.object(meta, "fit_calibrator", return_value=None), patch.object(meta, "apply_utility", side_effect=lambda r, q, m: r), patch("joblib.dump"):
                result = meta.train_nested_oof(records, questions, groups_path, splits, None, [], None, Path(folder) / "oof.csv", "sig", True)
            self.assertEqual(len(result), 50)
            self.assertEqual(len(fits), 31)  # 5 outer + 25 inner + one deployment model.
            for fold, (tr, va) in enumerate(splits):
                self.assertEqual(fits[fold * 6], set(map(int, tr)))
                for training in fits[fold * 6 + 1:fold * 6 + 6]:
                    self.assertTrue(training <= set(map(int, tr)))
                    self.assertFalse(training & set(map(int, va)))

    def test_real_lightgbm_query_group_api(self):
        records, gold = [], {}
        for i in range(50):
            matrix = np.zeros((3, len(meta.FEATURES)), np.float32)
            matrix[:, 0] = [0, 1, 2]
            ids = [i * 3 + j for j in range(3)]
            records.append({"qid": str(i), "meta_candidates": ids, "meta_features": matrix})
            gold[str(i)] = {"_G": [ids[-1]], "_w": 1}
        model = meta.fit_ranker(records, range(50), gold)
        scores = model.predict(records[0]["meta_features"])
        self.assertGreater(scores[2], scores[0])
        self.assertEqual(model.n_features_in_, len(meta.FEATURES))

    def test_resume_repairs_incomplete_tail_without_regeneration(self):
        posts = pd.DataFrame([{"post_id": 11, "date": "2024-01-01", "text": "Мост открыт", "channel": "A", "channel_type": "Паблик"}])
        questions = pd.DataFrame([{"qid": "q", "question": "Открыт ли мост?", "asked_at": "2024-01-02", "answer_format": "yesno"}])
        class FakeReader:
            generations = 0
            def __init__(self, cfg):
                self.model = None
            def prompt(self, row, indices, retriever):
                return json.dumps(row, ensure_ascii=False), {11: {"date": "2024-01-01", "text": "Мост открыт"}}
            def generate_batch(self, users):
                FakeReader.generations += len(users)
                return [json.dumps({"answer": "yes", "confidence": 0.8, "evidence_ids": [11]}) for _ in users]
        with tempfile.TemporaryDirectory() as folder:
            cfg = s.Config(cache=Path(folder), lexical_only=True, verify="none")
            retriever = s.Retriever(posts, cfg)
            item = {"qid": "q", "candidates": [0], "retrieval": [0], "max_lex": 1, "max_sem": -1,
                    "meta_candidates": [11], "meta_features": np.zeros((1, len(meta.FEATURES)), np.float32)}
            log = Path(folder) / "raw.jsonl"
            with patch.object(s, "Reader", FakeReader):
                first = s.infer(questions, [item], retriever, log, "sig")
                with log.open("a", encoding="utf-8") as stream:
                    stream.write('{"qid":')
                again = s.infer(questions, [item], retriever, log, "sig", resume=True)
                self.assertEqual(first, again)
                self.assertEqual(FakeReader.generations, 1)
                self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 1)
                changed = questions.copy()
                changed.loc[0, "question"] = "Другой вопрос?"
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    s.infer(changed, [item], retriever, log, "sig", resume=True)


if __name__ == "__main__":
    unittest.main()
