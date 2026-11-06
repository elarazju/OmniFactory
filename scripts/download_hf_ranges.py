"""Verified, resumable range download for a single pinned large Hub LFS file."""

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--filename", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partial", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    a = p.parse_args()
    assert len(a.revision) == 40 and 1 <= a.workers <= 16
    info = HfApi(endpoint=a.endpoint).model_info(a.repo, revision=a.revision, files_metadata=True)
    assert info.sha == a.revision
    item = next(f for f in info.siblings if f.rfilename == a.filename)
    assert item.lfs and item.size > 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        with a.output.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item.lfs.sha256
        print("Existing verified file", flush=True)
        return
    chunks = a.output.parent / ".ranges" / item.lfs.sha256
    chunks.mkdir(parents=True, exist_ok=True)
    unit = 256 * 1024 * 1024
    spans = [(start, min(start + unit, item.size)) for start in range(0, item.size, unit)]
    if a.partial:
        available = a.partial.stat().st_size
        with a.partial.open("rb") as src:
            for start, end in spans:
                if end > available:
                    break
                target = chunks / str(start)
                if target.exists() and target.stat().st_size == end - start:
                    continue
                src.seek(start)
                with target.open("wb") as dst:
                    left = end - start
                    while left:
                        block = src.read(min(left, 1024 * 1024))
                        assert block
                        dst.write(block)
                        left -= len(block)

    def fetch(span):
        start, end = span
        target = chunks / str(start)
        if target.exists() and target.stat().st_size == end - start:
            return end - start
        temporary = chunks / (str(start) + ".tmp")
        for attempt in range(4):
            try:
                url = f"{a.endpoint}/{a.repo}/resolve/{a.revision}/{a.filename}?download=true&range_start={start}"
                with requests.get(
                    url,
                    headers={"Range": f"bytes={start}-{end - 1}"},
                    stream=True,
                    timeout=(30, 60),
                ) as response:
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert (
                        response.headers.get("Content-Range")
                        == f"bytes {start}-{end - 1}/{item.size}"
                    )
                    with temporary.open("wb") as dst:
                        for block in response.iter_content(1024 * 1024):
                            dst.write(block)
                assert temporary.stat().st_size == end - start
                os.replace(temporary, target)
                return end - start
            except (requests.RequestException, AssertionError):
                if attempt == 3:
                    raise RuntimeError(f"Range {start}:{end} failed after four attempts") from None
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        completed = 0
        for future in as_completed([pool.submit(fetch, span) for span in spans]):
            completed += future.result()
            print(f"completed_bytes={completed} total_bytes={item.size}", flush=True)
    temporary = a.output.with_suffix(a.output.suffix + ".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as dst:
        for start, _ in spans:
            with (chunks / str(start)).open("rb") as src:
                while block := src.read(8 * 1024 * 1024):
                    digest.update(block)
                    dst.write(block)
    assert digest.hexdigest() == item.lfs.sha256, "assembled LFS hash mismatch"
    os.replace(temporary, a.output)
    shutil.rmtree(chunks)
    print(f"Verified SHA256 {item.lfs.sha256}", flush=True)


if __name__ == "__main__":
    main()
