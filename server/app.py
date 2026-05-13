"""FastAPI app. Endpoints: see API.md."""
import asyncio
import functools
import gzip
import json
import logging
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
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from passlib.hash import argon2
from pydantic import BaseModel, Field

from delphi.clients import OpenRouter
from delphi.scorers.classifier.prompts.detection_prompt import prompt as detection_prompt

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
N_FEATURES = 16384 

SCORER_MODEL = "anthropic/claude-haiku-4.5"
N_NEIGHBORS = 5
K_ITER = 5
N_SHOWN = 5
SCORE_TIMEOUT_S = 60

logger = logging.getLogger("taoci.score")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

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


def _preview(e: dict, n: int = 200) -> str:
    return "".join(e["tokens"])[:n].replace("\n", "\\n")


class _LoggingOpenRouter(OpenRouter):
    async def generate(self, prompt, **kw):
        logger.info("[LLM prompt]\n%s", json.dumps(prompt, ensure_ascii=False, indent=2))
        r = await super().generate(prompt, **kw)
        logger.info("[LLM response] %s", r.text if r else None)
        return r


async def _classify_batch(client, explanation: str,
                          batch: list[tuple[dict, bool, str]]) -> list[bool | None]:
    """One LLM call over a 5-example batch. All-or-nothing parse."""
    examples_str = "\n".join(f"Example {i}: {''.join(e['tokens'])}"
                             for i, (e, _, _) in enumerate(batch))
    logger.info("[Batch truth] %s",
                [("ACT" if a else f"DIS({o})") for _, a, o in batch])
    resp = await client.generate(detection_prompt(examples=examples_str,
                                                  explanation=explanation))
    text = resp.text if resp else ""
    m = re.search(r"\[.*?\]", text)
    if not m:
        return [None] * len(batch)
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return [None] * len(batch)
    return [bool(x) for x in raw] if len(raw) == len(batch) else [None] * len(batch)


async def score_explanation(fid: int, description: str) -> dict:
    logger.info("=== Scoring feature %d | description=%r ===", fid, description)
    P = top_activations(fid)
    logger.info("Positives P (%d):\n%s", len(P),
                "\n".join(f"  [{i}] {_preview(e)}" for i, e in enumerate(P)))

    neighbors = [n for n in _feature(fid)["neighbors"]
                 if n["idx"] != fid and n["idx"] < N_FEATURES][:N_NEIGHBORS]
    D: list[tuple[int, dict]] = []
    for n in neighbors:
        exs = top_activations(n["idx"])
        logger.info("Neighbor %d (cos=%s) -> %d examples:\n%s",
                    n["idx"], n.get("cos"), len(exs),
                    "\n".join(f"  [{i}] {_preview(e)}" for i, e in enumerate(exs)))
        D.extend((n["idx"], e) for e in exs)
    random.Random(42).shuffle(D)
    subpools = [D[i * N_SHOWN:(i + 1) * N_SHOWN] for i in range(K_ITER)]

    client = _LoggingOpenRouter(SCORER_MODEL, api_key=OPENROUTER_API_KEY, temperature=0)
    TP = TN = POS = NEG = 0
    bal_log: list[float | None] = []
    for i, sub in enumerate(subpools):
        items = ([(e, True, "P") for e in P]
                 + [(e, False, f"N{nidx}") for nidx, e in sub])
        random.Random(42 + i).shuffle(items)
        logger.info("=== iter %d ===", i)
        calls = [items[:N_SHOWN], items[N_SHOWN:]]
        preds = [await _classify_batch(client, description, c) for c in calls]
        if any(all(p is None for p in ps) for ps in preds):
            logger.warning("[iter %d] dropped (parse failure)", i)
            bal_log.append(None)
            continue
        tp_i = tn_i = vpos = vneg = 0
        for call, ps in zip(calls, preds):
            for (_, actual, _), pred in zip(call, ps):
                if pred is None:
                    continue
                if actual:
                    vpos += 1; tp_i += int(pred)
                else:
                    vneg += 1; tn_i += int(not pred)
        TP += tp_i; TN += tn_i; POS += vpos; NEG += vneg
        bal_i = 0.5 * (tp_i / max(vpos, 1) + tn_i / max(vneg, 1))
        bal_log.append(bal_i)
        logger.info("[iter %d] tp=%d/%d tn=%d/%d bal=%.3f",
                    i, tp_i, vpos, tn_i, vneg, bal_i)

    if POS < 25 or NEG < 25:
        raise HTTPException(500, f"too few parseable selections (POS={POS} NEG={NEG})")
    bal = 0.5 * (TP / POS + TN / NEG)
    score = max(0.0, 2 * bal - 1)
    logger.info("Result fid=%d TP=%d/%d TN=%d/%d bal=%.3f score=%.3f bal_per_iter=%s",
                fid, TP, POS, TN, NEG, bal, score,
                [None if b is None else round(b, 3) for b in bal_log])
    return {"score": score, "scorer_model_id": SCORER_MODEL}


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


app.mount("/", StaticFiles(directory=WEB_DIR, html=True, follow_symlink=True), name="web")
