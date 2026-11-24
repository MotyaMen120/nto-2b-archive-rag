"""Run full grouped OOF, train evidence ranker/calibrator, infer test and package files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "output_2b_final")
    parser.add_argument("--reader", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reader-batch", type=int, default=2)
    parser.add_argument("--wait-for-weights", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_for_weights:
        waiting_since = time.monotonic()
        manifest_path = root / "weights_2b" / "manifest.json"
        while True:
            try:
                prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                prepared = {}
            if {"encoder", "reranker"} <= set(prepared):
                break
            if time.monotonic() - waiting_since > 8 * 3600:
                raise TimeoutError("Weight preparation did not complete within eight hours; inference has not started.")
            print("Waiting for verified encoder/reranker weights...", flush=True)
            time.sleep(30)
    for name in ["encoder", "reranker"]:
        directory = root / "weights_2b" / name
        if not (directory / "config.json").exists() or not any(directory.glob("*.safetensors")):
            raise FileNotFoundError(f"Weights missing: {directory}. Run prepare_weights_2b.py before this offline command.")
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    command = [sys.executable, "-X", "utf8", str(root / "solution_2b_v2.py"),
               "--data-dir", str(root), "--encoder", str(root / "weights_2b" / "encoder"),
               "--reranker", str(root / "weights_2b" / "reranker"), "--reader", args.reader,
               "--input-tokens", "4096", "--output-tokens", "512", "--reader-batch", str(args.reader_batch),
               "--cache", str(root / ".cache_2b_final"), "--resume"]
    oof = args.output_dir / "oof.csv"
    test = args.output_dir / "submission.csv"
    started = time.perf_counter()
    subprocess.run(command + ["--mode", "val", "--train-meta", "--calibrate", "--output", str(oof)], env=environment, check=True, cwd=root)
    subprocess.run(command + ["--mode", "test", "--meta-ranker", str(oof.with_suffix(".meta.joblib")),
                              "--calibrator", str(oof.with_suffix(".calibrator.joblib")), "--output", str(test)], env=environment, check=True, cwd=root)
    oof_report = json.loads(oof.with_suffix(".report.json").read_text(encoding="utf-8"))
    test_report = json.loads(test.with_suffix(".report.json").read_text(encoding="utf-8"))
    result = {"complete_local_run": True, "oof_metric": oof_report["official_metric"],
              "raw_oof_metric": oof_report["raw_official_metric"], "test_rows": test_report["rows"],
              "resources_oof": oof_report["resources"], "resources_test": test_report["resources"],
              "test_cold_time_estimate_seconds": test_report["elapsed_seconds"] + oof_report["dense_build_seconds"],
              "total_elapsed_seconds": time.perf_counter() - started, "t4_verified": False,
              "note": "Measured on the recorded local GPU. T4 performance has not been measured. Weight downloads precede offline inference."}
    (args.output_dir / "completion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [root / name for name in ["solution_2b_v2.py", "meta_2b.py", "prepare_weights_2b.py", "run_complete_2b.py", "requirements_2b_v2.txt", "README_2b_v2.txt", "AGENTS.md"]]
    for pattern in ["score*.py", "submission*.py", "units*.py", "corpus*.parquet", "train*.csv", "test*.csv", "sample_submission*.csv"]:
        files.extend(root.glob(pattern))
    files.extend((root / "tests_2b_v2").glob("*.py"))
    files.extend(p for p in args.output_dir.glob("*") if p.is_file() and p.suffix != ".jsonl")
    manifest = root / "weights_2b" / "manifest.json"
    if manifest.exists():
        files.append(manifest)
    archive = root / "solution_2b_complete.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in dict.fromkeys(files):
            bundle.write(path, path.relative_to(root))
    print(json.dumps({"archive": str(archive), "submission": str(test), "S": result["oof_metric"]["S"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
