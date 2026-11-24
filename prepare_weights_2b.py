"""Network-enabled preparation ONLY. Inference never imports this script."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import time


MODELS = {
    "encoder": ("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"),
    "reranker": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
}


def parallel_file(repo, revision, filename, size, checksum, destination, workers):
    """Resumable HTTP ranges, with the publisher's full-file SHA256 checked at the end."""
    import requests
    if destination.exists() and destination.stat().st_size == size:
        digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
        if digest == checksum:
            return
        raise ValueError(f"Checksum mismatch in existing {destination}")
    block_size = 32 * 1024 * 1024
    pieces = destination.parent / ".parts" / filename
    pieces.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(size, start + block_size)) for start in range(0, size, block_size)]
    def fetch(bounds):
        start, stop = bounds
        part = pieces / f"{start:012d}.part"
        for attempt in range(6):
            offset = part.stat().st_size if part.exists() else 0
            if offset == stop - start:
                return
            if offset > stop - start:
                raise ValueError(f"Oversized preparation chunk: {part}")
            try:
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
                with requests.get(url, headers={"Range": f"bytes={start + offset}-{stop - 1}"}, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    expected = f"bytes {start + offset}-{stop - 1}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the requested byte range.")
                    with part.open("ab") as stream:
                        for data in response.iter_content(256 * 1024):
                            if stream.tell() + len(data) > stop - start:
                                raise ValueError("Server returned too many bytes.")
                            stream.write(data)
                if part.stat().st_size == stop - start:
                    return
            except (requests.RequestException, OSError):
                if attempt == 5:
                    raise
                time.sleep(min(10, 1 + attempt))
        raise RuntimeError(f"Incomplete preparation chunk {part}")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(fetch, bounds) for bounds in ranges}
        while pending:
            done, pending = wait(pending, timeout=15)
            for future in done:
                future.result()
            current = sum(p.stat().st_size for p in pieces.glob("*.part"))
            elapsed = time.monotonic() - started
            print(f"{filename}: {current / 2**20:.1f}/{size / 2**20:.1f} MiB ({current / 2**20 / max(1, elapsed):.2f} MiB/s)", flush=True)
    digest = hashlib.sha256()
    incoming = destination.with_name(destination.name + ".verified-incoming")
    with incoming.open("wb") as stream:
        for start, _ in ranges:
            with (pieces / f"{start:012d}.part").open("rb") as part:
                for data in iter(lambda: part.read(8 * 1024 * 1024), b""):
                    stream.write(data)
                    digest.update(data)
    if digest.hexdigest() != checksum:
        raise ValueError(f"Downloaded checksum mismatch: {destination}")
    os.replace(incoming, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("weights_2b"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    manifest = {}
    for name, (repo, revision) in MODELS.items():
        info = api.model_info(repo, revision=revision, files_metadata=True)
        files = [f.rfilename for f in info.siblings]
        has_safetensors = any(f.endswith(".safetensors") and "/" not in f for f in files)
        patterns = ["*.json", "tokenizer.model", "sentencepiece.bpe.model", "*.txt", "*.safetensors" if has_safetensors else "pytorch_model*.bin"]
        destination = args.output_dir / name
        print(f"Preparing {repo}@{revision} -> {destination}", flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        large = [file for file in info.siblings if "/" not in file.rfilename and file.size and file.size >= 8 * 1024 * 1024
                 and (file.rfilename.endswith(".safetensors") or file.rfilename == "tokenizer.json")]
        for file in large:
            checksum = file.lfs.sha256 if file.lfs else None
            if checksum is None:
                raise ValueError(f"Missing published checksum for {file.rfilename}")
            parallel_file(repo, revision, file.rfilename, file.size, checksum, destination / file.rfilename, args.workers)
        snapshot_download(repo, revision=revision, local_dir=destination, allow_patterns=patterns,
                          ignore_patterns=["onnx/*", "openvino/*", *[file.rfilename for file in large]], max_workers=3)
        manifest[name] = {"repo": repo, "revision": revision, "directory": str(destination),
                          "files": {p.name: p.stat().st_size for p in destination.glob("*") if p.is_file()}}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Weights prepared. Run inference with network disabled.", flush=True)


if __name__ == "__main__":
    main()
