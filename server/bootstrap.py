"""One-shot deploy prep. Idempotent.

Reads `DATABASE_URL` and `NEURONPEDIA_API_KEY` from .env (or environment).

    python -m server.bootstrap
"""
import gzip
import json
import os
import secrets
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from passlib.hash import argon2

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "np-l20-res-16k"
FEATURES_DIR = DATA_DIR / "features"
UMAP_BIN = ROOT / "web" / "umap.bin"

HF_REPO = "google/gemma-scope-2b-pt-res"
HF_FILE = "layer_20/width_16k/average_l0_71/params.npz"

NP_MODEL = "gemma-2-2b"
NP_LAYER = "20-gemmascope-res-16k"
NP_API = "https://www.neuronpedia.org/api/feature/{m}/{l}/{i}"
NP_N_FEATURES = 1000
NP_CONCURRENCY = 8
NP_EXPL_KEEP = ("description", "explanationModelName", "scoreV1", "scoreV2", "scores")

SEED_USER = "neuronpedia"
SEED_CREATED_AT = "epoch"  # 1970-01-01 sentinel: row predates the game
LABEL_MAX = 1000


def apply_schema() -> None:
    url = os.environ["DATABASE_URL"]
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql)
    print(f"[schema] applied to {url.rsplit('@', 1)[-1]}")


def _trim(raw: dict, idx: int) -> dict:
    by_bin: dict[tuple, list] = {}
    for a in raw.get("activations") or []:
        k = (a.get("binMin"), a.get("binMax"), a.get("binContains"))
        by_bin.setdefault(k, []).append({
            "tokens": a.get("tokens"),
            "values": a.get("values"),
            "max": a.get("maxValue"),
        })
    buckets = [
        {"binMin": k[0], "binMax": k[1], "binContains": k[2],
         "count": len(v), "examples": v}
        for k, v in by_bin.items()
    ]
    neighbors = [
        {"idx": i, "cos": c}
        for i, c in zip(raw.get("topkCosSimIndices") or [],
                        raw.get("topkCosSimValues") or [])
    ]
    return {
        "index": str(idx),
        "explanations": [{k: e.get(k) for k in NP_EXPL_KEEP}
                         for e in (raw.get("explanations") or [])],
        "buckets": buckets,
        "neighbors": neighbors,
    }


def _fetch_feature(idx: int, key: str, retries: int = 5) -> dict:
    req = urllib.request.Request(NP_API.format(m=NP_MODEL, l=NP_LAYER, i=idx),
                                 headers={"x-api-key": key})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return _trim(json.loads(r.read()), idx)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if getattr(e, "code", None) in (400, 401, 403, 404) and attempt == 0:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on feature {idx}")


def fetch_features() -> None:
    key = os.environ.get("NEURONPEDIA_API_KEY")
    if not key:
        raise SystemExit("NEURONPEDIA_API_KEY required")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(NP_N_FEATURES)
            if not (FEATURES_DIR / f"{i}.json.gz").exists()]
    if not todo:
        print(f"[features] all {NP_N_FEATURES} present")
        return
    print(f"[features] fetching {len(todo)}/{NP_N_FEATURES}", flush=True)
    t0 = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=NP_CONCURRENCY) as ex:
        futs = {ex.submit(_fetch_feature, i, key): i for i in todo}
        for f in as_completed(futs):
            i = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                print(f"  feature {i}: {e}", file=sys.stderr, flush=True)
                n_err += 1
                continue
            path = FEATURES_DIR / f"{i}.json.gz"
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as g:
                g.write(json.dumps(rec))
            tmp.rename(path)
            n_ok += 1
            if n_ok % 200 == 0:
                print(f"  {n_ok}/{len(todo)} ({time.time() - t0:.1f}s)", flush=True)
    print(f"[features] done {n_ok} OK / {n_err} err ({time.time() - t0:.1f}s)")


def build_umap_bin(seed: int = 0) -> None:
    import umap
    with tempfile.TemporaryDirectory() as td:
        path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, cache_dir=td)
        with np.load(path) as npz:
            W = npz["W_dec"].copy()
    emb = umap.UMAP(metric="cosine", random_state=seed).fit_transform(W)
    UMAP_BIN.parent.mkdir(parents=True, exist_ok=True)
    UMAP_BIN.write_bytes(emb.astype("<f4").tobytes())
    print(f"[umap] {emb.shape} -> {UMAP_BIN} ({UMAP_BIN.stat().st_size} B)")


def seed_submissions() -> None:
    files = sorted(FEATURES_DIR.glob("*.json.gz"))
    if not files:
        print(f"[seed] {FEATURES_DIR} empty, skip")
        return
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn, \
         conn.cursor() as cur:
        cur.execute("select id from profiles where username = %s", (SEED_USER,))
        row = cur.fetchone()
        if row:
            uid = row[0]
        else:
            uid = uuid.uuid4()
            cur.execute(
                "insert into profiles (id, username, password_hash) values (%s, %s, %s)",
                (str(uid), SEED_USER, argon2.hash(secrets.token_hex(32))),
            )
        cur.execute("delete from submissions where user_id = %s", (str(uid),))

        rows = []
        for path in files:
            with gzip.open(path, "rt") as f:
                rec = json.loads(f.read())
            fid = int(rec["index"])
            for e in rec.get("explanations") or []:
                label = (e.get("description") or "").strip()[:LABEL_MAX]
                if not label:
                    continue
                rows.append((str(uid), fid, label))
        if not rows:
            print("[seed] no explanations found")
            return
        cur.executemany(
            "insert into submissions (user_id, feature_id, label, created_at) "
            f"values (%s, %s, %s, '{SEED_CREATED_AT}')", rows,
        )
        print(f"[seed] inserted {len(rows)} submissions as {SEED_USER}")


if __name__ == "__main__":
    apply_schema()
    fetch_features()
    build_umap_bin()
    seed_submissions()
