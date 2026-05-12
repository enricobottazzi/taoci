"""FastAPI app. Endpoints: see API.md."""
import asyncio
import functools
import gzip
import json
import os
import random
import re
import secrets
import time
import uuid
from datetime import timezone
from pathlib import Path

import jwt
import psycopg
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from passlib.hash import argon2
from pydantic import BaseModel, Field

from delphi.clients import OpenRouter
from delphi.latents import (ActivatingExample, Latent, LatentRecord,
                            NonActivatingExample)
from delphi.scorers.classifier.detection import DetectionScorer
from transformers import AutoTokenizer

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
JWT_SECRET = os.environ["JWT_SECRET"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
JWT_ALG = "HS256"
JWT_TTL = 7 * 24 * 3600

USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
FEATURES_DIR = Path(__file__).resolve().parent.parent / "np-l20-res-16k" / "features"
TOP_K = 5
N_FEATURES = 1000  # TODO: raise to 16384 once full Neuronpedia dump is fetched (see bootstrap.NP_N_FEATURES)

SCORER_MODEL = "anthropic/claude-sonnet-4.5"
SCORER_MODULE = "blocks.20.hook_resid_post"
SCORER_TOKENIZER = "unsloth/gemma-2-2b"
N_TEST = 20
N_DISTRACTORS = 20
N_SHOWN = 5
SCORE_TIMEOUT_S = 60

app = FastAPI()
bearer = HTTPBearer(auto_error=False)


def db():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("set time zone 'UTC'")
    return conn


def mint_token(user_id: uuid.UUID, username: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": str(user_id), "username": username, "iat": now, "exp": now + JWT_TTL},
        JWT_SECRET, algorithm=JWT_ALG,
    )


def require_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(401, "missing bearer token")
    try:
        return jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"invalid token: {e}")


class AuthIn(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=2, max_length=256)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model_loaded": True}


@app.post("/auth")
def auth(body: AuthIn) -> dict:
    if not USERNAME_RE.match(body.username):
        raise HTTPException(400, "username must be 2-32 chars [A-Za-z0-9_-]")
    with db() as conn, conn.cursor() as cur:
        cur.execute("select id, password_hash from profiles where username = %s",
                    (body.username,))
        row = cur.fetchone()
        if row is None:
            uid = uuid.uuid4()
            cur.execute(
                "insert into profiles (id, username, password_hash) values (%s, %s, %s)",
                (str(uid), body.username, argon2.hash(body.password)),
            )
        else:
            uid, pw_hash = row
            if not argon2.verify(body.password, pw_hash):
                raise HTTPException(401, "wrong password")
    return {"access_token": mint_token(uid, body.username),
            "user": {"id": str(uid), "username": body.username}}


@app.get("/map")
def map_data(_: dict = Depends(require_user)) -> dict:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            select username, count(*)::int
            from feature_best group by username order by count(*) desc
        """)
        leaderboard = [{"username": u, "features_led": n}
                       for u, n in cur.fetchall()]
        cur.execute("""
            select feature_id, username, label, score, found_at
            from feature_best
        """)
        features = [{"id": fid, "user": u, "label": lbl,
                     "score": float(s) if s is not None else None,
                     "found_at": fa.astimezone(timezone.utc).date().isoformat()}
                    for fid, u, lbl, s, fa in cur.fetchall()]
    return {"leaderboard": leaderboard, "features": features}


@functools.lru_cache(maxsize=2048)
def _feature(fid: int) -> dict:
    with gzip.open(FEATURES_DIR / f"{fid}.json.gz", "rt") as f:
        return json.load(f)


def _top_examples(fid: int) -> list[dict]:
    return next(b for b in _feature(fid)["buckets"]
                if b["binContains"] == -1)["examples"]


def top_activations(fid: int) -> list[dict]:
    seen, out = set(), []
    for e in _top_examples(fid):
        key = "\0".join(e["tokens"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"tokens": e["tokens"], "values": e["values"], "max": e["max"]})
        if len(out) == TOP_K:
            break
    return out


@functools.lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(SCORER_TOKENIZER)


def _to_example(e: dict, cls):
    tok = _tokenizer()
    return cls(tokens=torch.tensor(tok.convert_tokens_to_ids(e["tokens"])),
               activations=torch.tensor(e["values"]),
               str_tokens=e["tokens"])


async def score_explanation(fid: int, description: str) -> dict:
    test = [_to_example(e, ActivatingExample)
            for e in _top_examples(fid)[:N_TEST]]
    pool: list[dict] = []
    for n in _feature(fid)["neighbors"]:
        if n["idx"] == fid or n["idx"] >= N_FEATURES:
            continue
        pool.extend(_top_examples(n["idx"]))
        if len(pool) >= N_DISTRACTORS * 3:
            break
    random.Random(42).shuffle(pool)
    not_active = [_to_example(e, NonActivatingExample)
                  for e in pool[:N_DISTRACTORS]]

    record = LatentRecord(latent=Latent(SCORER_MODULE, fid), test=test,
                          not_active=not_active, explanation=description)
    scorer = DetectionScorer(
        client=OpenRouter(SCORER_MODEL, api_key=OPENROUTER_API_KEY),
        n_examples_shown=N_SHOWN, verbose=False)
    outs = (await scorer(record)).score or []
    if not outs:
        raise HTTPException(500, "scorer returned no parseable selections")
    pos = sum(o.activating for o in outs)
    neg = len(outs) - pos
    tp = sum(o.correct for o in outs if o.activating)
    tn = sum(o.correct for o in outs if not o.activating)
    bal = 0.5 * (tp / max(pos, 1) + tn / max(neg, 1))
    return {"score": bal, "scorer_model_id": SCORER_MODEL}


@app.get("/play")
def play(_: dict = Depends(require_user)) -> dict:
    fid = secrets.randbelow(N_FEATURES)
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            select username, label, score, found_at
            from feature_best where feature_id = %s
        """, (fid,))
        row = cur.fetchone()
    best = None
    if row is not None:
        u, lbl, s, fa = row
        best = {"user": u, "label": lbl,
                "score": float(s) if s is not None else None,
                "found_at": fa.astimezone(timezone.utc).date().isoformat()}
    return {"id": fid, "top_activations": top_activations(fid), "best": best}


class ScoreIn(BaseModel):
    feature_id: int = Field(ge=0, lt=N_FEATURES)
    description: str = Field(min_length=1, max_length=1000)


@app.post("/score")
async def score(body: ScoreIn, u: dict = Depends(require_user)) -> dict:
    try:
        r = await asyncio.wait_for(
            score_explanation(body.feature_id, body.description),
            timeout=SCORE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, "scorer timed out")
    with db() as conn, conn.cursor() as cur:
        cur.execute("select max(score) from submissions where feature_id = %s",
                    (body.feature_id,))
        prev = cur.fetchone()[0]
        cur.execute(
            "insert into submissions"
            " (user_id, feature_id, label, score, scorer_model_id)"
            " values (%s, %s, %s, %s, %s) returning submission_id",
            (u["sub"], body.feature_id, body.description,
             r["score"], r["scorer_model_id"]),
        )
        sid = cur.fetchone()[0]
    return {"submission_id": sid, "score": r["score"],
            "is_new_best": prev is None or r["score"] > prev}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
