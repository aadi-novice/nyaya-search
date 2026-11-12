"""Download the AWS Open Data 'Indian Supreme Court Judgments' bucket (public, no credentials).

Downloads metadata first (small), then English tars, then regional tars.
Resumable: files whose local size matches the S3 size are skipped; partial
downloads are written to *.part and restarted.

Usage: python scripts/download_sc.py [--dest docs] [--workers 4] [--skip-regional]
"""
import argparse
import concurrent.futures as cf
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_objects(prefix):
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        for attempt in range(1, 6):
            try:
                root = ET.fromstring(urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q), timeout=60).read())
                break
            except Exception as e:  # noqa: BLE001
                print(f"[list retry {attempt}/5] {prefix}: {e}", flush=True)
                time.sleep(3 * attempt)
        else:
            raise RuntimeError(f"listing failed for {prefix}")
        for c in root.findall(NS + "Contents"):
            yield c.find(NS + "Key").text, int(c.find(NS + "Size").text)
        token = root.findtext(NS + "NextContinuationToken")
        if not token:
            return


def fetch(key, size, dest, retries=5):
    path = os.path.join(dest, *key.split("/"))
    if os.path.exists(path) and os.path.getsize(path) == size:
        return key, "skip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(BUCKET + urllib.parse.quote(key), timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if os.path.getsize(part) != size:
                raise IOError(f"size mismatch {os.path.getsize(part)} != {size}")
            os.replace(part, path)
            return key, "ok"
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt}/{retries}] {key}: {e}", flush=True)
            time.sleep(2 * attempt)
    return key, "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="docs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-regional", action="store_true")
    a = ap.parse_args()

    groups = [("metadata/parquet/", None), ("metadata/json/", None), ("metadata/tar/", None),
              ("data/tar/", "/english/")]
    if not a.skip_regional:
        groups.append(("data/tar/", "/regional/"))

    failed = []
    for prefix, must_contain in groups:
        objs = [(k, s) for k, s in list_objects(prefix) if must_contain is None or must_contain in k]
        total = sum(s for _, s in objs)
        print(f"== {prefix} {must_contain or ''}: {len(objs)} files, {total / 1e9:.2f} GB", flush=True)
        done_bytes = 0
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(fetch, k, s, a.dest): s for k, s in objs}
            for fut in cf.as_completed(futs):
                key, status = fut.result()
                done_bytes += futs[fut]
                if status == "FAILED":
                    failed.append(key)
                if status != "skip":
                    print(f"{status:6} {done_bytes / 1e9:7.2f}/{total / 1e9:.2f} GB  {key}", flush=True)
    print("FAILED:", failed if failed else "none", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
