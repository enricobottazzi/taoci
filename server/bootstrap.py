"""One-shot deploy prep: schema + dataset sync + umap.bin. Idempotent.

    DATABASE_URL=... TAOCI_DATA_DIR=./np-l20-res-16k python -m server.bootstrap
"""
import os
from pathlib import Path

import boto3
import numpy as np
import psycopg
from botocore import UNSIGNED
from botocore.config import Config
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TAOCI_DATA_DIR", ROOT / "np-l20-res-16k"))
UMAP_BIN = ROOT / "web" / "umap.bin"

S3_BUCKET = "neuronpedia-datasets"
S3_PREFIX = "v1/gemma-2-2b/20-gemmascope-res-16k"
HF_REPO = "google/gemma-scope-2b-pt-res"
HF_FILE = "layer_20/width_16k/average_l0_71/params.npz"


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


if __name__ == "__main__":
    apply_schema()
    sync_dataset()
    build_umap_bin()
