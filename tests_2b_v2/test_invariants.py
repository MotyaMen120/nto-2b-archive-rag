import json
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import solution_2b_v2 as s


def corpus():
    return pd.DataFrame([
        {"post_id": 11, "date": "2024-02-01", "channel": "A", "channel_type": "vk", "text": "Открытие моста запланировано на весну. Стоимость 40 млн руб."},
        {"post_id": 12, "date": "2024-02-02", "channel": "B", "channel_type": "tg", "text": "Открытие моста запланировано на весну. Стоимость 40 млн руб."},
        {"post_id": 13, "date": "2024-02-03", "channel": "A", "channel_type": "vk", "text": "Открытие моста запланировано на весну. Стоимость 40 млн руб."},
        {"post_id": 14, "date": "2024-04-01", "channel": "C", "channel_type": "tg", "text": "Открытие моста запланировано на весну. Стоимость 40 млн руб."},
        {"post_id": 15, "date": "2024-05-01", "channel": "D", "channel_type": "tg", "text": "Мост открыли. Стоимость 90 млн руб."},
    ])


def row(**kwargs):
    return {"qid": "q1", "question": "Сколько дней прошло между сообщениями?", "asked_at": "2024-04-01", "answer_format": "number", "unit": "дней", **kwargs}


class Invariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.cfg = s.Config(cache=self.path, lexical_only=True)
        self.retriever = s.Retriever(corpus(), self.cfg)

    def tearDown(self):
        self.retriever.embeddings = self.retriever.parents = self.retriever.first = None
        gc.collect()
        self.tmp.cleanup()

    def test_future_cannot_change_bm25_statistics_or_results(self):
        prefix = corpus().iloc[:2]
        past = s.Retriever(prefix, self.cfg)
        full = self.retriever
        a = past.lexical("мост стоимость", past.cutoff("2024-02-02"))
        b = full.lexical("мост стоимость", full.cutoff("2024-02-02"))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(past.search(["мост"], "2024-02-02"), full.search(["мост"], "2024-02-02"))
        self.assertEqual(full.search(["мост"], "2023-12-31")[0], [])

    def test_dense_never_reads_future_chunks(self):
        class GuardedEmbeddings:
            def __getitem__(self, key):
                if key.stop > 2:
                    raise AssertionError("Future chunks were scored.")
                return np.asarray([[1, 0], [0, 1]], dtype=np.float16)[key]
        retriever = self.retriever
        retriever.embeddings = GuardedEmbeddings()
        retriever.parents = np.arange(5, dtype=np.int32)
        retriever.query_cache["мост"] = np.asarray([1, 0], dtype=np.float32)
        np.testing.assert_array_equal(retriever.semantic("мост", 2), [1, 0])
        # Exercise the tensor scoring branch with poison future rows, on CPU to
        # avoid competing for VRAM with a separately running inference job.
        import torch
        retriever.gpu_chunks = torch.tensor([[1, 0], [0, 1], [float("nan"), 0], [float("nan"), 0], [float("nan"), 0]], dtype=torch.float16)
        np.testing.assert_array_equal(retriever.semantic("мост", 2), [1, 0])

    def test_dense_cache_is_memmap_and_rejects_same_length_changed_corpus(self):
        model_dir = self.path / "encoder"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        self.cfg.encoder = str(model_dir)
        calls = []
        def load(retriever):
            retriever.encoder_model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))
        def encode(retriever, texts, output=None):
            calls.append(len(texts))
            output[:] = [1, 0]
            return output
        with patch.object(s.Retriever, "load_encoder", load), patch.object(s.Retriever, "encode", encode):
            self.retriever.build_dense()
            self.assertIsInstance(self.retriever.embeddings, np.memmap)
            self.assertEqual(self.retriever.embeddings.dtype, np.float16)
            again = s.Retriever(corpus(), self.cfg)
            again.build_dense()
            self.assertEqual(len(calls), 1)
            changed = corpus()
            changed.loc[0, "text"] = "Другая новость, тот же размер корпуса."
            s.Retriever(changed, self.cfg).build_dense()
            self.assertEqual(len(calls), 2)

    def test_evidence_reposts_require_time_and_other_channel(self):
        r = self.retriever
        self.assertTrue(r.similar_repost(0, 1)[0])
        self.assertFalse(r.similar_repost(0, 2)[0])
        self.assertFalse(r.similar_repost(0, 3)[0])
        self.assertFalse(r.similar_repost(0, 4)[0])
        self.assertEqual(r.evidence([11, 14], [0, 3, 1, 2])[:3], [11, 14, 12])

    def test_context_is_chronological_and_deduplicated(self):
        selected = self.retriever.context([3, 1, 0, 2], "Статус моста")
        self.assertEqual(selected, sorted(selected))
        self.assertFalse(0 in selected and 1 in selected)

    def test_new_file_autodetection(self):
        (self.path / "train (2).csv").touch()
        (self.path / "train (5).csv").touch()
        self.assertEqual(s.find_file(self.path, "train.csv").name, "train (5).csv")
        (self.path / "train.csv").touch()
        self.assertEqual(s.find_file(self.path, "train.csv").name, "train.csv")
        self.assertEqual(s.find_file(self.path, "train.csv", self.path / "train (2).csv").name, "train (2).csv")

    def test_groups_are_strictly_disjoint_and_required(self):
        questions = pd.DataFrame([row(qid=f"q{i}", question=f"История {i}", evidence=str(i + 100)) for i in range(10)])
        path = self.path / "train_groups.csv"
        # This is a synthetic fixture, never a replacement for the competition groups.
        pd.DataFrame({"qid": questions.qid, "group": [str(i // 2) for i in range(10)]}).to_csv(path, index=False)
        folds, splits = s.group_folds(questions, path)
        self.assertEqual(set(folds), set(range(5)))
        for a, b in zip(folds[::2], folds[1::2]):
            self.assertEqual(a, b)
        questions.loc[9, "evidence"] = questions.loc[0, "evidence"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            s.group_folds(questions, path)

    def test_python_date_arithmetic_and_publication_sources(self):
        visible = {11: {"date": "2024-01-31", "text": "Анонс"}, 14: {"date": "2024-03-30", "text": "Открыли"}}
        expr = {"op": "days_diff", "a": {"value": "2024-01-31", "post_id": 11, "source": "publication_date"},
                "b": {"value": "2024-03-30", "post_id": 14, "source": "publication_date"}}
        self.assertEqual(s.calculate(expr, visible, row()), "59")
        expr["op"] = "full_months_diff"
        self.assertEqual(s.calculate(expr, visible, row()), "1")
        expr["b"]["value"] = "2024-03-31"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            s.calculate(expr, visible, row())

    def test_russian_dates_and_invalid_dates(self):
        self.assertEqual(s.extract_dates("с 29 февраля 2024 до 30.03.2024; 2024-02-30"), ["2024-02-29", "2024-03-30"])

    def test_number_units_and_percentages(self):
        self.assertEqual(s.normalize_answer("204.4", "number", "млрд руб", "млн руб"), "204400")
        self.assertEqual(s.normalize_answer("100", "number", "тыс. руб", "руб"), "100000")
        self.assertEqual(s.number_string("0.00001"), "0.00001")
        visible = {11: {"text": "Цена выросла с 100 до 125 рублей."}}
        expr = {"op": "pct_change", "a": {"value": "100", "post_id": 11, "quote": "с 100 до 125"},
                "b": {"value": "125", "post_id": 11, "quote": "с 100 до 125"}}
        self.assertEqual(s.calculate(expr, visible, row(unit="%")), "25")
        expr["b"]["value"] = "999"
        with self.assertRaisesRegex(ValueError, "absent"):
            s.calculate(expr, visible, row(unit="%"))

    def test_convert_is_applied_once(self):
        visible = {11: {"text": "Контракт на 204,4 млрд руб."}}
        expr = {"op": "convert", "unit": "млн руб", "a": {"value": "204.4", "unit": "млрд руб", "post_id": 11, "quote": "204,4 млрд руб"}}
        self.assertEqual(s.calculate(expr, visible, row(unit="млн руб")), "204400")

    def test_safe_json_parser_computes_even_without_numeric_answer(self):
        visible = {11: {"date": "2024-02-01", "text": "Анонс"}, 12: {"date": "2024-02-02", "text": "Открыли"}}
        reply = {"answer": "вычислит Python", "evidence_ids": [999, 11], "confidence": 0.8,
                 "math_expr": {"op": "days_diff", "a": {"value": "2024-02-01", "post_id": 11, "source": "publication_date"},
                               "b": {"value": "2024-02-02", "post_id": 12, "source": "publication_date"}}}
        parsed = s.parse_reply(json.dumps(reply), row(), visible)
        self.assertEqual(parsed["answer"], "1")
        self.assertEqual(parsed["cited"], [11, 12])
        visible[13] = {"date": "2024-02-01", "text": "Другой пост"}
        reply["evidence_ids"] = [13, 12, 11]
        self.assertEqual(s.parse_reply(json.dumps(reply), row(), visible)["cited"][:2], [11, 12])

    def test_uncomputed_arithmetic_and_invented_sources_abstain(self):
        reply = {"answer": "42", "confidence": 0.9}
        self.assertEqual(s.parse_reply(json.dumps(reply), row(), {})["answer"], "skip")
        reply["math_expr"] = None
        self.assertEqual(s.parse_reply(json.dumps(reply), row(), {})["answer"], "skip")
        self.assertEqual(s.parse_reply("not json", row(), {})["answer"], "skip")
        reply["math_expr"] = {"op": "sub", "a": {"value": "50", "post_id": 999, "quote": "50"}, "b": {"value": "8", "post_id": 999, "quote": "8"}}
        self.assertEqual(s.parse_reply(json.dumps(reply), row(), {})["answer"], "skip")

    def test_promised_event_is_not_replaced_with_a_calculated_plan(self):
        visible = {11: {"date": "2024-02-01", "text": "Открытие запланировано на 15 марта 2024 года."}}
        reply = {"answer": "promised", "evidence_ids": [11], "confidence": 0.9,
                 "math_expr": {"op": "days_diff", "a": {"value": "2024-02-01", "post_id": 11, "source": "publication_date"},
                               "b": {"value": "2024-03-15", "post_id": 11, "source": "text", "quote": "15 марта 2024 года"}}}
        parsed = s.parse_reply(json.dumps(reply), row(asked_at="2024-02-10"), visible)
        self.assertEqual(parsed["answer"], "promised")

    def test_calibration_trains_only_on_other_groups(self):
        questions = pd.DataFrame([row(qid=f"q{i}", question=f"Вопрос {i}", evidence="11", answer="2", level="light") for i in range(10)])
        records = [{"qid": f"q{i}", "answer": "2", "evidence": [11], "fallback_evidence": [11, 12],
                    "confidence": i / 10, "max_lex": 2, "max_sem": -1, "context": [11], "parse_error": ""} for i in range(10)]
        scorer = s.module_from_file(s.find_file(s.ROOT, "score.py"))
        gold = [{"id": f"q{i}", "answer_format": "number", "answer": "2", "_G": [11], "_w": 1, "level": "light"} for i in range(10)]
        groups = self.path / "groups.csv"
        pd.DataFrame({"qid": questions.qid, "group": [str(i // 2) for i in range(10)]}).to_csv(groups, index=False)
        _, splits = s.group_folds(questions.assign(evidence=[str(100 + i) for i in range(10)]), groups)
        fitted, predicted = [], []
        class TrackingRegressor:
            def __init__(self, **kwargs):
                pass
            def fit(self, x, y, sample_weight=None):
                fitted.append(set(x[:, 0]))
                return self
            def predict(self, x):
                predicted.append(set(x[:, 0]))
                return np.full(len(x), -0.1)
        with patch("sklearn.ensemble.RandomForestRegressor", TrackingRegressor), patch("joblib.dump"):
            result = s.calibrate_oof(records, questions, splits, scorer, gold, self.path / "calibrator.joblib", "signature")
        self.assertEqual(len(fitted), 6)
        for train, val in zip(fitted[:5], predicted):
            self.assertFalse(train & val)
        self.assertTrue(all(r["answer"] == "skip" and r["evidence"] == [11, 12] for r in result))

    def test_official_scorer_skip_really_ignores_evidence(self):
        scorer = s.module_from_file(s.find_file(s.ROOT, "score.py"))
        item = {"id": "q", "answer_format": "number", "answer": "1", "_G": [11], "_w": 1, "level": "light"}
        result = scorer.score_item(item, {"answer": "skip", "evidence": [11]})
        self.assertEqual(result["e"], 0)
        self.assertAlmostEqual(result["s"], -0.175)

    def test_transport_temporal_checks_and_official_cli(self):
        validator = s.find_file(s.ROOT, "submission.py")
        questions = pd.DataFrame([row(question="Есть ли мост?", answer_format="yesno")])
        records = [{"qid": "q1", "answer": "skip", "evidence": [11, 12]}]
        checked = s.validate_records(records, questions, self.retriever, s.module_from_file(validator))
        self.assertEqual(checked[0]["evidence"], "11 12")
        questions_path = self.path / "questions.csv"
        # Question fixture is not a generated submission; its schema is checked by the CLI.
        questions.to_csv(questions_path, index=False)
        corpus_path = self.path / "corpus.parquet"
        corpus().to_parquet(corpus_path, index=False)
        s.write_submission(records, questions, self.retriever, self.path / "submission.csv", validator, corpus_path, questions_path)
        records[0]["evidence"] = [15]
        with self.assertRaisesRegex(ValueError, "future"):
            s.validate_records(records, questions, self.retriever, s.module_from_file(validator))
        records[0].update(answer="no_info", evidence=[11])
        with self.assertRaisesRegex(ValueError, "no_info"):
            s.validate_records(records, questions, self.retriever, s.module_from_file(validator))


if __name__ == "__main__":
    unittest.main()
