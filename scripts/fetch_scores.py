#!/usr/bin/env python3
"""Fetch per-(feature, explanation) scores from Neuronpedia.

Hits GET /api/feature/{model}/{layer}/{index} for every feature and dumps the
attached `explanations[*]` (each with its `scores: [...]`) as gzipped JSONL,
batched 1024 features per file to mirror the S3 layout. Resumable: existing
output batches are skipped.

Set NEURONPEDIA_API_KEY in a `.env` file at the repo root (or in your env).
Get a key at https://neuronpedia.org/account.
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://www.neuronpedia.org/api/feature/{m}/{l}/{i}"
KEEP_EXP = ("id", "description", "explanationModelName", "typeName",
            "scoreV1", "scoreV2", "scores", "triggeredByUser")


def fetch(model: str, layer: str, idx: int, key: str, retries: int = 5) -> dict:
    req = urllib.request.Request(API.format(m=model, l=layer, i=idx),
                                 headers={"x-api-key": key})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                n = json.loads(r.read())
                return {"index": str(idx),
                        "explanations": [{k: e.get(k) for k in KEEP_EXP}
                                         for e in (n.get("explanations") or [])]}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code in (400, 401, 403, 404) and attempt == 0:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on feature {idx}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-2-2b")
    p.add_argument("--layer", default="20-gemmascope-res-16k")
    p.add_argument("--n-features", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="np-l20-res-16k/explanation-scores")
    a = p.parse_args()

    key = os.environ.get("NEURONPEDIA_API_KEY")
    if not key:
        sys.exit("set NEURONPEDIA_API_KEY (https://neuronpedia.org/account)")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    batches = [(b, range(b * a.batch_size,
                         min((b + 1) * a.batch_size, a.n_features)))
               for b in range((a.n_features + a.batch_size - 1) // a.batch_size)]

    for b, rng in batches:
        path = out / f"batch-{b}.jsonl.gz"
        if path.exists():
            print(f"[{b+1}/{len(batches)}] skip {path}", flush=True); continue
        print(f"[{b+1}/{len(batches)}] fetching {len(rng)} features "
              f"(idx {rng.start}..{rng.stop - 1}, concurrency={a.concurrency})",
              flush=True)
        rows: dict[int, dict] = {}
        errors = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = {ex.submit(fetch, a.model, a.layer, i, key): i for i in rng}
            for n, f in enumerate(as_completed(futs), 1):
                i = futs[f]
                try:
                    rows[i] = f.result()
                except Exception as e:
                    errors += 1
                    print(f"  feature {i}: {e}", file=sys.stderr, flush=True)
                if n % 50 == 0 or n == len(rng):
                    rate = n / (time.time() - t0)
                    print(f"  [{b+1}/{len(batches)}] {n}/{len(rng)} "
                          f"({rate:.1f}/s, errors={errors})", flush=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as g:
            for i in sorted(rows):
                g.write(json.dumps(rows[i]) + "\n")
        tmp.rename(path)
        print(f"[{b+1}/{len(batches)}] wrote {path}  "
              f"({len(rows)}/{len(rng)} features, {time.time() - t0:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
