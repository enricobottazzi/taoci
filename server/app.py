"""FastAPI app. Endpoints: see API.md."""
import os
import re
import time
import uuid

import jwt
import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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


@app.get("/")
@app.get("/index.html")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/login.html")
def login() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "login.html"))


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
