"""Regression checks for batch-dependent quantized activation scales."""
import argparse
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cpu_semantic_2b as semantic


class BatchSensitiveEngine:
    def __init__(self, directory, threads, length):
        pass

    def encode(self, inputs, batch_size, dense):
        if not inputs:
            return np.array([])
        lengths = np.array([len(str(x)) for x in inputs],np.float32)
        # Dynamic quantization of a real batch can couple the scores across inputs.
        factor = lengths.max()
        if not dense:
            return lengths/factor
        vectors = np.stack([lengths/factor, np.ones_like(lengths)],axis=-1)
        return vectors/np.linalg.norm(vectors,axis=1,keepdims=True)


class TemporalBatches(unittest.TestCase):
    def run_case(self, root, include_future):
        root.mkdir()
        weights = root/"weights"
        weights.mkdir()
        for filename in ("tokenizer.json","model.onnx"):
            (weights/filename).write_bytes(b"fixture")
        docs = [
            {"post_id":0,"date":date(2024,1,1),"text":"Мост в городе","channel":"a"},
            {"post_id":1,"date":date(2024,1,1),"text":"Строят мост в городе","channel":"b"},
            {"post_id":2,"date":date(2024,1,3),"text":"Новые подробности "*80,"channel":"c"},
        ]
        qs = [
            {"qid":"late","asked_at":"2024-01-03","question":"Мост построен?","options":"","answer_format":"yesno"},
            {"qid":"early","asked_at":"2024-01-01","question":"Мост построен?","options":"","answer_format":"yesno"},
        ]
        if not include_future:
            docs = docs[:2]
            qs = qs[1:]
        corpus = pd.DataFrame(docs)
        corpus_path = root/"corpus.parquet"
        corpus.to_parquet(corpus_path)
        seed = [{"qid":q["qid"],"ids":[0,1,2] if q["qid"]=="late" else [0,1]} for q in qs]
        semantic.lexical.dump_jsonl(root/"seed_test.jsonl",seed)
        args = argparse.Namespace(output=root,cache_dir=None,threads=1,dense_weights=weights,
                                  ce_weights=weights,e5_weights=weights,top=5,with_e5=False)
        frames = {"test":pd.DataFrame(qs)}
        with patch.object(semantic,"Engine",BatchSensitiveEngine):
            semantic.dense_search(args,corpus,frames,corpus_path,root,"dense")
            semantic.cross_encode(args,corpus,frames,corpus_path,root)
        return [semantic.read_ranks(root/name)["early"] for name in ("dense_test.jsonl","ce_test.jsonl")]

    def test_later_posts_and_later_question_cannot_change_earlier_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(self.run_case(root/"prefix",False),self.run_case(root/"full",True))


if __name__ == "__main__":
    unittest.main()
