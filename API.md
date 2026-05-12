# taoci — deployment spec

Three boxes:

```
┌──────────┐   HTTPS + JWT    ┌────────────────────┐   DATABASE_URL   ┌──────────────┐
│ frontend │ ───────────────► │ scoring server     │ ───────────────► │ Postgres     │
│ (static) │                  │ (FastAPI)          │                  │              │
└──────────┘                  └─────────┬──────────┘                  └──────────────┘
                                        │ reads
                                        ▼
                              static dataset on disk
```

- **Frontend** is dumb: only talks to the scoring server.
- **Scoring server** is the only thing that holds DB credentials. Every read and write goes through it.
- **Postgres** is any managed or self-hosted instance reachable via `DATABASE_URL`. The frontend never opens a direct connection.
- **Static dataset** (UMAP, activations) ships with the server.
---

## 1. Static data (read-only, immutable per release)

| path | size | who reads | notes |
| --- | --- | --- | --- |
| `web/umap.bin` | ~130 KB | FE | `Float32Array(N, 2)` star positions for `/map` (only FE-fetched static asset) |
| `np-l20-res-16k/activations/batch-*.jsonl.gz` | ~314 MB (16 files) | BE only | positives for `/play` (`top_activations`) and `/score`, distractors for `/score` |
| `np-l20-res-16k/features/batch-*.jsonl.gz` | ~14 MB (16 files) | BE only | `topkCosSimIndices` (precomputed neighbour list) for `/score` |

---

## 2. Server API

FastAPI. Holds `DATABASE_URL`, `JWT_SECRET`, `OPENROUTER_API_KEY`. Protected routes require `Authorization: Bearer <jwt>`; the server validates the JWT locally (HS256 over `JWT_SECRET`) and reads `sub` as the user id.

### `GET /healthz`  *(public)*

**Role**: liveness probe. Returns `{ ok: true, model_loaded: true }` once the tokenizer is warm. No auth, no DB.

### `POST /auth`  *(public)*

**Role**: the only entry point for users. Single endpoint handles both signup and login because the PoC login form is shared.

**Request**: `{ username, password }`
**Response**: `{ access_token, user: { id, username } }`

### `GET /map`  *(auth)*

**Role**: hydrate map.html in **one round trip**. Called every time the user lands on the map.

**Response**:
```json
{
  "leaderboard": [{ "username": "neo", "features_led": 1287 }],
  "features": [{ "id": 877, "user": "neo", "label": "references to municipal recycling", "score": 0.84, "found_at": "2025-03-12" }]
}
```

### `GET /play`  *(auth)*

**Role**: hydrate play.html in one round trip. Server picks a random feature on every call. The page should never know which id ahead of time. Click PLAY → land on a fresh feature.

**Response**:
```json
{
  "id": 14035,
  "top_activations": [{ "tokens": [...], "values": [...], "max": 0.84 }],
  "best": { "user": "apoc", "label": "...", "score": 0.053, "found_at": "..." },
}
```

`best` may be null (undiscovered, or caller has never tried this one).

### `POST /score`  *(auth)*

**Role**: the actual game move. User submits an explanation; server runs the LLM scorer and writes the result.

**Request**: `{ feature_id, description }`
**Response**: `{ submission_id, score, is_new_best }`

---

Errors are uniform: `{ "error": "<code>", "message": "<human>" }`. 401 (auth), 400 (validation), 404 (unknown feature/user), 429 (rate-limit), 504 (scorer timeout), 500 (everything else).

---

## 3. Database (Postgres)

Two tables + one view. Passwords are hashed with argon2id (`passlib[argon2]`) and stored on `profiles`; JWTs are issued and verified by the server itself.

```sql
create extension if not exists citext;
create extension if not exists pgcrypto;  -- for gen_random_uuid()

-- one row per user
create table profiles (
  id             uuid primary key default gen_random_uuid(),
  username       citext unique not null check (char_length(username) between 2 and 32),
  password_hash  text not null,
  created_at     timestamptz not null default now()
);

-- append-only event log: one row per /score call
create table submissions (
  submission_id    bigserial primary key,
  user_id          uuid not null references profiles(id) on delete cascade,
  feature_id       int  not null check (feature_id between 0 and 16383),
  label            text not null check (char_length(label) between 1 and 500),
  score            real not null check (score between 0 and 1),
  scorer_model_id  text not null,                 -- e.g. 'anthropic/claude-sonnet-4.5'
  created_at       timestamptz not null default now()
);

create index submissions_feature_score_desc
  on submissions (feature_id, score desc, created_at asc);
create index submissions_user_idx on submissions (user_id);

-- current best per feature, derived from submissions
create view feature_best as
select distinct on (s.feature_id)
  s.feature_id,
  s.submission_id,
  s.user_id,
  p.username,
  s.label,
  s.score,
  s.scorer_model_id,
  s.created_at as found_at
from submissions s
join profiles p on p.id = s.user_id
order by s.feature_id, s.score desc, s.created_at asc;  -- earliest tie-break
```

Endpoint → query map:

| endpoint | query |
| --- | --- |
| `POST /auth` | `select id, password_hash from profiles where username=$1`; verify argon2id, else `insert into profiles ...`. Issue HS256 JWT with `sub=id`. |
| `GET /play` | random `feature_id` → `select * from feature_best where feature_id = $1`. |
| `GET /map` features | `select feature_id, username, label, score, found_at from feature_best`. |
| `GET /map` leaderboard | `select username, count(*) features_led from feature_best group by username order by features_led desc`. |
| `POST /score` | `insert into submissions ...`; `is_new_best` = `not exists (select 1 from submissions where feature_id=$1 and submission_id<>$new and score>=$score)`. |

---
