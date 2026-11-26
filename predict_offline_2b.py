"""Rebuild indexes and predict with a frozen, unpacked NTO 2B bundle."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_member(directory, member):
    path = (directory / member).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Missing or invalid bundle member: {member}")
    return path


def input_file(explicit, name, bundle):
    places = [Path.cwd(), bundle]
    if explicit:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit.resolve()
    for directory in places:
        if (directory / name).is_file():
            return (directory / name).resolve()
        p = Path(name)
        suffixes = []
        for file in directory.glob(f"{p.stem} (*){p.suffix}"):
            suffix = file.stem[len(p.stem) + 2:-1]
            if file.is_file() and suffix.isdigit():
                suffixes.append((int(suffix), file))
        if suffixes:
            return max(suffixes)[1].resolve()
    raise FileNotFoundError(f"Supply --{Path(name).stem} for {name}.")


def check_dates(path, questions, corpus):
    import pyarrow.parquet as pq
    with questions.open(encoding="utf-8-sig", newline="") as stream:
        asked = {r["qid"]: r["asked_at"] for r in csv.DictReader(stream)}
    dates = {}
    for batch in pq.ParquetFile(corpus).iter_batches(batch_size=8192, columns=["post_id", "date"]):
        for pid, day in zip(batch[0].to_pylist(), batch[1].to_pylist()):
            dates[str(pid)] = str(day)[:10]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        count = 0
        for row in csv.DictReader(stream):
            evidence = row["evidence"].split()
            if any(dates[pid] > asked[row["qid"]] for pid in evidence):
                raise ValueError(f"Future evidence in {row['qid']}")
            if row["answer"] == "no_info" and evidence:
                raise ValueError(f"Evidence on no_info in {row['qid']}")
            count += 1
    return count


def publish(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
        temporary = Path(target.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--work-dir", type=Path, help="Parent directory for a unique, temporary cold-run cache")
    parser.add_argument("--profile", type=Path, help="Save cold-run timings and sampled process-tree RSS as JSON")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--train", type=Path, help="Optionally rescore the packaged OOF predictions on supplied train")
    parser.add_argument("--groups", type=Path, help="Verify the training groups fingerprint, if supplied")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf8"))
    if manifest["schema"] != 1:
        raise ValueError("Unsupported bundle schema")
    for member, sha in manifest["sha256"].items():
        if digest(checked_member(bundle, member)) != sha:
            raise ValueError(f"Bundle checksum mismatch: {member}")
    config = manifest["pipeline"]
    corpus = input_file(args.corpus, "corpus.parquet", bundle)
    test = input_file(args.test, "test.csv", bundle)
    output = args.output.resolve()
    protected = {corpus, test, bundle / "manifest.json", *(bundle / k for k in manifest["sha256"])}
    if output in protected or (args.profile and args.profile.resolve() in protected | {output}):
        raise ValueError("Output and profile paths must be distinct from inputs and packaged assets")
    if args.groups:
        group_hash = manifest["training_sha256"].get("groups")
        if not group_hash or digest(args.groups) != group_hash:
            raise ValueError("Training groups differ from the packaged provenance")
    if args.train and "train" in manifest["training_sha256"]:
        if digest(args.train) != manifest["training_sha256"]["train"]:
            raise ValueError("Train file differs from packaged OOF provenance")
    threads = config["threads"] if args.threads is None else args.threads
    if threads < 1:
        raise ValueError("--threads must be positive")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    if config.get("nli"):
        env["NTO2B_NLI_DIR"] = str(bundle / "weights_2b/nli")
    else:
        env.pop("NTO2B_NLI_DIR", None)
    from psutil import Process, Error
    own = Process()
    profile = {"stages": {}, "cold_cache": True, "device": "CPU", "threads": threads}
    def run(stage, script, *argv):
        print(f"[{stage}]", flush=True)
        cmd = [sys.executable, str(bundle / script), *map(str, argv)]
        begin = time.monotonic()
        peak = 0
        with subprocess.Popen(cmd, cwd=bundle, env=env) as child:
            while child.poll() is None:
                rss = 0
                for p in [own, *own.children(recursive=True)]:
                    try:
                        rss += p.memory_info().rss
                    except Error:
                        pass
                peak = max(peak, rss)
                time.sleep(.2)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, cmd)
        profile["stages"][stage] = {"seconds": time.monotonic()-begin, "peak_rss_bytes": peak}
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    common = ["--test", test, "--corpus", corpus]
    with tempfile.TemporaryDirectory(prefix="nto2b-replay-", dir=args.work_dir) as temporary:
        work = Path(temporary).resolve()
        lex = work / "lexical"
        rank = lex
        run("lexical", "cpu_rank_2b.py", *common, "--model", bundle / "models/lexical.joblib",
            "--threads", threads, "--output", lex)
        if config["semantic"]:
            rank = work / "semantic"
            semantic = [*common, "--model", bundle / "models/semantic.joblib", "--rank-dir", lex,
                        "--output", rank, "--dense-weights", bundle / "weights_2b/cpu_dense",
                        "--ce-weights", bundle / "weights_2b/cpu_ce", "--top", config["top"],
                        "--threads", threads]
            stages = ["dense"]
            if config["e5"]:
                semantic.extend(["--with-e5", "--e5-weights", bundle / "weights_2b/cpu_e5"])
                stages.append("e5")
            for stage in [*stages, "ce", "fuse"]:
                run(stage, "cpu_semantic_2b.py", stage, *semantic)
        if config.get("story"):
            first = rank
            rank = work / "story"
            run("story_evidence", "cpu_story_2b.py", *common, "--model", bundle / "models/story.joblib",
                "--lex-dir", lex, "--semantic-dir", first, "--weights-dir", bundle / "weights_2b",
                "--output", rank, "--threads", threads)
        reader = work / "reader/submission.csv"
        run("reader", "run_cpu_2b.py", *common, "--model", bundle / "models/readers.joblib",
            "--rank-dir", rank, "--output", reader)
        final = reader
        if config["policy"]:
            final = work / "policy/submission.csv"
            run("abstention", "cpu_abstention_2b.py", *common, "--model", bundle / "models/policy.joblib",
                "--input", reader.parent, "--seed-dir", lex, "--rank-dir", rank, "--output", final)
        run("official_validation", "submission.py", final, "--questions", test, "--corpus", corpus)
        profile["questions"] = check_dates(final, test, corpus)
        publish(final, output)
    if args.train:
        run("official_oof_score", "score.py", bundle / "validation/oof.csv", "--gold", args.train.resolve(),
            "--boot", "0")
    profile["sampled_peak_rss_bytes"] = max(stage["peak_rss_bytes"] for stage in profile["stages"].values())
    profile["submission_sha256"] = digest(output)
    profile["corpus_sha256"] = digest(corpus)
    profile["questions_sha256"] = digest(test)
    if (profile["questions_sha256"] == manifest["reference_test_sha256"] and
            profile["corpus_sha256"] == manifest["training_sha256"]["corpus"]):
        profile["matches_packaged_submission"] = (
            profile["submission_sha256"] == manifest["sha256"]["validation/submission.csv"])
        print(f"Exact packaged reference match: {profile['matches_packaged_submission']}", flush=True)
    profile["seconds"] = time.monotonic()-start
    if args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        args.profile.write_text(json.dumps(profile, indent=2), encoding="utf8")
    print(f"Validated {profile['questions']} questions in {profile['seconds']:.1f} seconds: {output}")


if __name__ == "__main__":
    main()
