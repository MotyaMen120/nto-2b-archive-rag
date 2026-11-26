"""Build a portable, CPU-only NTO 2B inference ZIP from selected frozen artifacts."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

from predict_offline_2b import check_dates, digest
import solution_2b_v2 as base


PUBLISHED = {
    "cpu_dense": {"repo": "Vades/rubert-tiny2-onnx-optim", "upstream": "cointegrated/rubert-tiny2",
                  "revision": "4c01a78b769b5bf9cc51b4d6ae7c95e5bf0bd352",
                  "model": "bert-onnx-optim/model_optimized.onnx", "tokenizer": "bert-onnx-optim/tokenizer.json"},
    "cpu_ce": {"repo": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
               "revision": "1427fd652930e4ba29e8149678df786c240d8825",
               "model": "onnx/model_quint8_avx2.onnx", "tokenizer": "tokenizer.json"},
    "cpu_e5": {"repo": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
               "model": "onnx/model_qint8_avx512_vnni.onnx", "tokenizer": "tokenizer.json"},
    "nli": {"repo": "cointegrated/rubert-tiny-bilingual-nli", "revision": "3907306f5e5d65e281a524c536883b766c89ded2",
            "model": "Local ONNX export of model.safetensors", "tokenizer": "tokenizer.json"},
}


def training_commands(config):
    data = '--corpus CORPUS --train TRAIN --groups GROUPS --test TEST'
    cmds = [f"python cpu_rank_2b.py {data} --threads {config['threads']} --output run/lexical"]
    rank = "run/lexical"
    if config["semantic"]:
        rank = "run/semantic"
        extra = " --with-e5" if config["e5"] else ""
        cmds.append(f"python cpu_semantic_2b.py all {data} --threads {config['threads']} --rank-dir run/lexical "
                    f"--output {rank} --top {config['top']}{extra}")
    if config.get("story"):
        # The baseline OOF is used solely for comparing the official score with fixed answers.
        cmds.append(f"python run_cpu_2b.py {data} --rank-dir {rank} --seed-dir run/lexical --output run/baseline/submission.csv")
        cmds.append(f"python cpu_story_2b.py {data} --threads {config['threads']} --lex-dir run/lexical "
                    f"--semantic-dir {rank} --fixed-answers run/baseline/oof.csv --output run/story")
        rank = "run/story"
    cmds.append(f"python run_cpu_2b.py {data} --rank-dir {rank} --seed-dir run/lexical --output run/reader/submission.csv")
    if config["policy"]:
        cmds.append(f"python cpu_abstention_2b.py {data} --input run/reader --rank-dir {rank} --seed-dir run/lexical "
                    "--output run/calibrated/submission.csv")
    return cmds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexical-model", type=Path, required=True)
    p.add_argument("--semantic-model", type=Path)
    p.add_argument("--story-model", type=Path, help="Optional fitted event/cross-channel reranker")
    p.add_argument("--reader-model", type=Path, required=True)
    p.add_argument("--policy-model", type=Path)
    p.add_argument("--with-e5", action="store_true")
    p.add_argument("--weights-dir", type=Path, default=base.ROOT / "weights_2b")
    p.add_argument("--nli-dir", type=Path, help="Optional ONNX folder for a reader trained with NLI")
    p.add_argument("--result-dir", type=Path, required=True, help="Selected run with oof.csv, report.json and submission.csv")
    p.add_argument("--top", type=int, default=32, help="Lexical candidates sent to the semantic cross-encoder")
    p.add_argument("--threads", type=int, default=4)
    for name in ("corpus", "test", "train", "groups"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.with_e5 and not a.semantic_model:
        p.error("--with-e5 needs a semantic ranker")
    if a.story_model and not a.with_e5:
        p.error("The event reranker uses both dense encoders and needs --with-e5")
    if a.top < 1 or a.threads < 1:
        p.error("--top and --threads must be positive")
    train = base.find_file(base.ROOT, "train.csv", a.train)
    test = base.find_file(base.ROOT, "test.csv", a.test)
    corpus = base.find_file(base.ROOT, "corpus.parquet", a.corpus)
    groups = base.find_file(base.ROOT, "train_groups.csv", a.groups)
    config = dict(semantic=bool(a.semantic_model), e5=a.with_e5, policy=bool(a.policy_model),
                  nli=bool(a.nli_dir), story=bool(a.story_model), top=a.top, threads=a.threads)
    names = ["predict_offline_2b.py", "package_cpu_2b.py", "solution_2b_v2.py", "run_cpu_2b.py",
             "cpu_rank_2b.py", "cpu_numeric_2b.py", "cpu_categorical_2b.py"]
    if config["semantic"]:
        names.append("cpu_semantic_2b.py")
    if config["story"]:
        names.append("cpu_story_2b.py")
    if config["policy"]:
        names.append("cpu_abstention_2b.py")
    if config["nli"]:
        names.append("cpu_nli_2b.py")
    # A saved reader can require NLI even if all inputs exist on this machine.
    import joblib
    reader = joblib.load(a.reader_model)
    if bool(reader.get("categorical", {}).get("nli_path")) != config["nli"]:
        p.error("Supply --nli-dir precisely for readers trained with NLI")
    files = {name: base.ROOT / name for name in names}
    for name in ("score.py", "submission.py", "units.py"):
        files[name] = base.find_file(base.ROOT, name)
    files.update({"models/lexical.joblib": a.lexical_model, "models/readers.joblib": a.reader_model,
                  "validation/oof.csv": a.result_dir / "oof.csv", "validation/report.json": a.result_dir / "report.json",
                  "validation/submission.csv": a.result_dir / "submission.csv"})
    if config["semantic"]:
        files["models/semantic.joblib"] = a.semantic_model
    if config["story"]:
        files["models/story.joblib"] = a.story_model
    if config["policy"]:
        files["models/policy.joblib"] = a.policy_model
    neural = (["cpu_dense", "cpu_ce"] if config["semantic"] else []) + (["cpu_e5"] if a.with_e5 else [])
    if a.nli_dir:
        neural.append("nli")
    for key in neural:
        folder = a.nli_dir if key == "nli" else a.weights_dir / key
        for file in ["model.onnx", "tokenizer.json", *(["config.json"] if key == "nli" else [])]:
            files[f"weights_2b/{key}/{file}"] = folder / file
    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(file)
    for prediction, questions in ((files["validation/oof.csv"],train),(files["validation/submission.csv"],test)):
        subprocess.run([sys.executable, str(files["submission.py"]), str(prediction), "--questions",str(questions),
                        "--corpus",str(corpus)], check=True)
        check_dates(prediction,questions,corpus)
    scorer = base.module_from_file(files["score.py"])
    items, measured = scorer.run(str(files["validation/oof.csv"]), [str(train)])
    report = json.loads(files["validation/report.json"].read_text(encoding="utf8"))
    if abs(measured["S"] - report["S"]) > 1e-7 or measured["invalid"]:
        raise ValueError("OOF predictions do not match the supplied report")
    frame = base.read_questions(train)
    folds, _ = base.group_folds(frame, groups)
    by_qid = {qid:int(fold) for qid,fold in zip(frame.qid,folds)}
    fold_s = [scorer.total([r for r in items if by_qid[r["qid"]] == fold]) for fold in range(5)]
    if any(abs(x-y) > 1e-7 for x,y in zip(fold_s,report.get("fold_S",fold_s))):
        raise ValueError("Group folds differ from the supplied OOF report")
    deps = ["numpy","pandas","pyarrow","scipy","scikit-learn","nltk","pymorphy3","pymorphy3-dicts-ru",
            "lightgbm","joblib","psutil","threadpoolctl"]
    if neural:
        deps.extend(["onnxruntime","tokenizers"])
    command = "python predict_offline_2b.py --bundle . --corpus CORPUS --test TEST --output submission.csv --profile replay.json"
    cmds = training_commands(config)
    notes = (f"# НТО 2B. Офлайн-решение\n\nГрупповая OOF-оценка на {measured['n']} вопросах "
             f"составила {measured['S']:.4f} из 100. Оценки по пяти фолдам "
             + ", ".join(f"{x:.2f}" for x in fold_s) + ".\n\n"
             f"## Запуск\n\nПроверенная версия Python — {platform.python_version()}. "
             "Установите пакеты из requirements.txt до запуска. Разархивируйте комплект, "
             "перейдите в его каталог и подставьте пути к корпусу и вопросам.\n\n"
             f"```text\n{command}\n```\n\n"
             "Каждый запуск собирает индекс в новой временной папке и проверяет CSV официальным валидатором. "
             "Параметр --work-dir указывает диск для временных индексов. "
             "Файл validation/submission.csv содержит предсказания для исходных 400 вопросов. "
             "Файл validation/oof.csv содержит предсказания на валидационных фолдах. "
             "В manifest.json приведены контрольные суммы файлов, ревизии весов и хеши использованных данных.\n\n"
             "## Воспроизведение обучения\n\nПодставьте пути TRAIN, GROUPS, TEST и CORPUS. "
             + ("Для обучения категориального модуля задайте переменную окружения NTO2B_NLI_DIR=weights_2b/nli. "
                if config["nli"] else "") +
             "Полная процедура использует GroupKFold из train_groups.csv.\n\n```text\n" + "\n".join(cmds) + "\n```\n")
    generated = {"requirements.txt": ("\n".join(f"{d}=={version(d)}" for d in deps)+"\n").encode(),
                 "README.md": notes.encode("utf8")}
    sha = {k:digest(v) for k,v in files.items()}
    sha.update({k:hashlib.sha256(v).hexdigest() for k,v in generated.items()})
    manifest = {"schema":1, "pipeline":config, "python":platform.python_version(), "sha256":sha,
                "published_weights":{key:PUBLISHED[key] for key in neural},
                "training_sha256":{k:digest(v) for k,v in dict(train=train,groups=groups,corpus=corpus).items()},
                "reference_test_sha256":digest(test), "oof":{"n":measured["n"], "S":measured["S"],"fold_S":fold_s},
                "training_commands":cmds}
    generated["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf8")
    destination = a.output.resolve()
    if destination in {path.resolve() for path in files.values()} | {p.resolve() for p in (train,test,corpus,groups)}:
        raise ValueError("The output archive must be separate from every input")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as temp:
        partial = Path(temp.name)
    try:
        with zipfile.ZipFile(partial,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for name,data in generated.items():
                archive.writestr(name,data)
            for name,path in files.items():
                archive.write(path,name)
        with zipfile.ZipFile(partial) as archive:
            if bad := archive.testzip():
                raise ValueError(f"Invalid ZIP member: {bad}")
        os.replace(partial,destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"{destination} ({destination.stat().st_size:,} bytes). Official grouped OOF {measured['S']:.4f}.")


if __name__ == "__main__":
    main()
