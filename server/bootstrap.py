"""One-shot deploy prep. Idempotent.

Reads `DATABASE_URL` from .env (or environment).

    python -m server.bootstrap
"""
import gzip
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import psycopg
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from passlib.hash import argon2

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "np-l20-res-16k"
EXPL_DIR = DATA_DIR / "explanation-scores"
UMAP_BIN = ROOT / "web" / "umap.bin"

S3_BUCKET = "neuronpedia-datasets"
S3_PREFIX = "v1/gemma-2-2b/20-gemmascope-res-16k"
HF_REPO = "google/gemma-scope-2b-pt-res"
HF_FILE = "layer_20/width_16k/average_l0_71/params.npz"

NP_MODEL = "gemma-2-2b"
NP_LAYER = "20-gemmascope-res-16k"
NP_API = "https://www.neuronpedia.org/api/feature/{m}/{l}/{i}"
NP_N_FEATURES = 16384
NP_BATCH_SIZE = 1024
NP_CONCURRENCY = 8
NP_KEEP = ("id", "description", "explanationModelName", "typeName",
           "scoreV1", "scoreV2", "scores", "triggeredByUser")

SEED_USER = "neuronpedia"
SEED_METHOD = "eleuther_recall"
LABEL_MAX = 1000


def apply_schema() -> None:
    url = os.environ["DATABASE_URL"]
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql)
    print(f"[schema] applied to {url.rsplit('@', 1)[-1]}")


def sync_dataset() -> None:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    for folder in ("activations", "features"):
        local = DATA_DIR / folder
        local.mkdir(parents=True, exist_ok=True)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/{folder}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                dst = local / Path(key).name
                if dst.exists() and dst.stat().st_size == obj["Size"]:
                    continue
                s3.download_file(S3_BUCKET, key, str(dst))
                print(f"[sync] {key} -> {dst}")
        print(f"[sync] {folder} OK ({sum(1 for _ in local.iterdir())} files)")


def build_umap_bin(seed: int = 0) -> None:
    if UMAP_BIN.exists():
        print(f"[umap] {UMAP_BIN} exists, skip")
        return
    import umap
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    W = np.load(path)["W_dec"]
    emb = umap.UMAP(metric="cosine", random_state=seed).fit_transform(W)
    UMAP_BIN.parent.mkdir(parents=True, exist_ok=True)
    UMAP_BIN.write_bytes(emb.astype("<f4").tobytes())
    print(f"[umap] {emb.shape} -> {UMAP_BIN} ({UMAP_BIN.stat().st_size} B)")


def _fetch_feature(idx: int, key: str, retries: int = 5) -> dict:
    req = urllib.request.Request(NP_API.format(m=NP_MODEL, l=NP_LAYER, i=idx),
                                 headers={"x-api-key": key})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                n = json.loads(r.read())
                return {"index": str(idx),
                        "explanations": [{k: e.get(k) for k in NP_KEEP}
                                         for e in (n.get("explanations") or [])]}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if getattr(e, "code", None) in (400, 401, 403, 404) and attempt == 0:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on feature {idx}")


def fetch_explanation_scores() -> None:
    key = os.environ.get("NEURONPEDIA_API_KEY")
    if not key:
        print("[explanations] NEURONPEDIA_API_KEY not set, skip"); return
    EXPL_DIR.mkdir(parents=True, exist_ok=True)
    n_batches = (NP_N_FEATURES + NP_BATCH_SIZE - 1) // NP_BATCH_SIZE
    for b in range(n_batches):
        path = EXPL_DIR / f"batch-{b}.jsonl.gz"
        if path.exists():
            continue
        rng = range(b * NP_BATCH_SIZE, min((b + 1) * NP_BATCH_SIZE, NP_N_FEATURES))
        print(f"[explanations] batch {b + 1}/{n_batches}: fetching {len(rng)} features", flush=True)
        rows: dict[int, dict] = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=NP_CONCURRENCY) as ex:
            futs = {ex.submit(_fetch_feature, i, key): i for i in rng}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    rows[i] = f.result()
                except Exception as e:
                    print(f"  feature {i}: {e}", file=sys.stderr, flush=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as g:
            for i in sorted(rows):
                g.write(json.dumps(rows[i]) + "\n")
        tmp.rename(path)
        print(f"[explanations] wrote {path} ({len(rows)}/{len(rng)}, "
              f"{time.time() - t0:.1f}s)", flush=True)


def seed_submissions() -> None:
    files = sorted(EXPL_DIR.glob("batch-*.jsonl.gz"))
    if not files:
        print(f"[seed] {EXPL_DIR} empty, skip"); return
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn, \
         conn.cursor() as cur:
        cur.execute("select id from profiles where username = %s", (SEED_USER,))
        row = cur.fetchone()
        if row:
            uid = row[0]
            cur.execute("select 1 from submissions where user_id = %s limit 1", (uid,))
            if cur.fetchone():
                print(f"[seed] {SEED_USER} submissions exist, skip"); return
        else:
            uid = uuid.uuid4()
            cur.execute(
                "insert into profiles (id, username, password_hash) values (%s, %s, %s)",
                (str(uid), SEED_USER, argon2.hash(secrets.token_hex(32))),
            )

        rows = []
        for path in files:
            with gzip.open(path, "rt") as f:
                for line in f:
                    rec = json.loads(line)
                    fid = int(rec["index"])
                    for e in rec.get("explanations") or []:
                        recalls = [s for s in (e.get("scores") or [])
                                   if s.get("explanationScoreTypeName") == SEED_METHOD
                                   and s.get("value") is not None]
                        if not recalls:
                            continue
                        best = max(recalls, key=lambda s: s["value"])
                        label = (e.get("description") or "").strip()[:LABEL_MAX]
                        if not label:
                            continue
                        rows.append((str(uid), fid, label, float(best["value"]),
                                     best.get("explanationScoreModelName") or "unknown"))
        if not rows:
            print(f"[seed] no {SEED_METHOD} scores found"); return
        cur.executemany(
            "insert into submissions (user_id, feature_id, label, score, scorer_model_id) "
            "values (%s, %s, %s, %s, %s)", rows,
        )
        print(f"[seed] inserted {len(rows)} submissions as {SEED_USER}")


if __name__ == "__main__":
    apply_schema()
    sync_dataset()
    build_umap_bin()
    fetch_explanation_scores()
    seed_submissions()
