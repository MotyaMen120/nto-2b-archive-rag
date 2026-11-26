"""Pre-inference download of a published model blob with resumable ranges and final SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import threading

import requests


def fetch_blob(url, target, size, sha, workers=12):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size==size:
        with target.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()==sha:
                return
        raise ValueError(f"Existing file differs from published SHA-256: {target}")
    parts=target.parent/".parts"/sha
    parts.mkdir(parents=True,exist_ok=True)
    block=4*1024*1024
    starts=list(range(0,size,block))
    local=threading.local()
    def get(start):
        path=parts/f"{start:012d}.part"
        stop=min(size,start+block)
        if not hasattr(local,"session"):
            local.session=requests.Session()
        n=0
        while n<25:
            offset=path.stat().st_size if path.exists() else 0
            if start+offset==stop:
                return
            if start+offset>stop:
                raise ValueError("Oversized downloaded range")
            left=start+offset
            # Change boundaries on repeated failures; some CDN routes stall at exact byte offsets.
            right=min(stop,left+max(192*1024,block//(2**min(n,4))))-1
            try:
                with local.session.get(url,headers={"Range":f"bytes={left}-{right}","Accept-Encoding":"identity"},
                                       params={"part":f"{left}-{right}","try":str(n)},stream=True,timeout=(9,24)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {left}-{right}/{size}":
                        raise ValueError("Server did not return the requested byte range")
                    with path.open("ab") as file:
                        for chunk in response.iter_content(64*1024):
                            if file.tell()+len(chunk)>stop-start:
                                raise ValueError("Response exceeds requested range")
                            file.write(chunk)
                n=0
            except (requests.RequestException,OSError,ValueError):
                n+=1
                time.sleep(min(4,.2*n))
        raise RuntimeError(f"Range still incomplete at {start}")
    def current():
        return sum(p.stat().st_size for p in parts.glob("*.part"))
    print(f"Resume {current()/2**20:.1f} / {size/2**20:.1f} MiB",flush=True)
    failures=[]
    tick=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task in as_completed([pool.submit(get,s) for s in starts]):
            try:
                task.result()
            except RuntimeError as e:
                failures.append(str(e))
            if time.monotonic()-tick>=15:
                print(f"Downloaded {current()/2**20:.1f} / {size/2**20:.1f} MiB",flush=True)
                tick=time.monotonic()
    if failures:
        raise RuntimeError(f"{len(failures)} incomplete ranges; rerun this command to resume")
    incoming=target.with_suffix(target.suffix+".verified-incoming")
    h=hashlib.sha256()
    with incoming.open("wb") as dest:
        for s in starts:
            with (parts/f"{s:012d}.part").open("rb") as file:
                for chunk in iter(lambda:file.read(1024*1024),b""):
                    dest.write(chunk)
                    h.update(chunk)
    if h.hexdigest()!=sha or incoming.stat().st_size!=size:
        raise ValueError("Full content hash does not match")
    incoming.replace(target)
    print(f"Verified {target}: {sha}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--size",type=int,required=True)
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    fetch_blob(args.url,args.output,args.size,args.sha256,args.workers)
