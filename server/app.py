"""FastAPI app. Endpoints: see API.md."""
import os
import re
import secrets
import time
import uuid

import jwt
import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from passlib.hash import argon2
from pydantic import BaseModel, Field

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALG = "HS256"
JWT_TTL = 7 * 24 * 3600

USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")

app = FastAPI()
bearer = HTTPBearer(auto_error=False)


def db():
    return psycopg.connect(DATABASE_URL, autocommit=True)


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
                     "found_at": fa.date().isoformat()}
                    for fid, u, lbl, s, fa in cur.fetchall()]
    return {"leaderboard": leaderboard, "features": features}


@app.get("/play")
def play(_: dict = Depends(require_user)) -> dict:
    fid = secrets.randbelow(16384)
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
                "found_at": fa.date().isoformat()}
    return {"id": fid, "best": best}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
