# taoci specs

```
                                                     DATABASE_URL      ┌──────────────────┐
                                                  ┌──────────────────► │ Postgres         │
                                                  │  (dynamic data)    │                  │
┌──────────┐   HTTPS + JWT    ┌───────────────────┴┐                   └──────────────────┘
│ frontend │ ───────────────► │ server             │
│ (static) │                  │ (FastAPI)          │                   ┌──────────────────┐
└──────────┘                  └───────────────────┬┘                   │ dataset on disk  │
                                                  │   reads            │                  │
                                                  └──────────────────► └──────────────────┘
                                                      (static data)
```

---

## 1. Server API

FastAPI. Holds `DATABASE_URL`, `JWT_SECRET`, `OPENROUTER_API_KEY` and `NEURONPEDIA_API_KEY` Protected routes require `Authorization: Bearer <jwt>`; the server validates the JWT locally (HS256 over `JWT_SECRET`) and reads `sub` as the user id.

### `GET /healthz`  *(public)*

**Role**: liveness probe. Returns `{ ok: true, model_loaded: true }`.

### `POST /auth`  *(public)*

**Role**: the only entry point for users. Single endpoint handles both signup and login.

**Request**: `{ username, password }`
**Response**: `{ access_token, user: { id, username } }`

### `GET /map`  *(auth)*

**Role**: hydrate map.html. Called every time the user lands on the map.

**Response**:
```json
{
  "leaderboard": [{ "username": "neo", "features_led": 1287 }],
  "features": [{ "id": 877, "user": "neo", "label": "references to municipal recycling", "score": 0.84, "found_at": "2025-03-12" }]
}
```

### `GET /play`  *(auth)*

**Role**: hydrate play.html. Server picks a random feature on every call.

**Response**:
```json
{
  "id": 14035,
  "top_activations": [{ "tokens": [...], "values": [...], "max": 0.84 }],
  "best": { "user": "apoc", "label": "...", "score": 0.053, "found_at": "..." },
}
```

### `POST /score`  *(auth)*

**Role**: the actual game move. User submits an explanation; server runs the LLM scorer and writes the result.

**Request**: `{ feature_id, description }` (`description`: 1–1000 chars)
**Response**: `{ submission_id, score, is_new_best }` (`score` ∈ [0, 1])

**Scoring procedure** — delphi's `DetectionScorer` (== eleuther_recall):

1. **Build evidence set** from `np-l20-res-16k/features/{feature_id}.json.gz`:
   - `test`: first `N_TEST=20` examples of the top bucket (`binContains == -1`) — these *do* activate the feature.
   - `not_active`: top-bucket examples of the feature's `neighbors[].idx` (highest cosine first, skipping self), pooled until `≥ 3·N_DISTRACTORS`, shuffled with `Random(42)`, truncated to `N_DISTRACTORS=20` — these are hard negatives that look semantically nearby but should *not* match the label.
2. **Tokenize** each example's `tokens` with `unsloth/gemma-2-2b` (cached, loaded once) into a `delphi.latents` `ActivatingExample` / `NonActivatingExample`.
3. **Score** with `DetectionScorer(client=OpenRouter("anthropic/claude-sonnet-4.5"), n_examples_shown=5)`: the LLM is shown groups of 5 examples + the user's `description` and must mark which ones activate.
4. **Reduce** to balanced accuracy `0.5 · (TP/POS + TN/NEG)` over all parseable responses; this is `score`.
5. **Persist** one row in `submissions` (`user_id`, `feature_id`, `label=description`, `score`, `scorer_model_id='anthropic/claude-sonnet-4.5'`, `created_at=now()`). `is_new_best = score > max(prior score for this feature_id)` (true if no prior real submission).

Hard timeout 60 s on the scorer call → 504. Scorer returning zero parseable selections → 500.

---

Errors are uniform: `{ "error": "<code>", "message": "<human>" }`. 401 (auth), 400 (validation), 404 (unknown feature/user), 429 (rate-limit), 504 (scorer timeout), 500 (everything else).

---

## 2. Static data (read-only, immutable per release)

| path | size | notes |
| --- | --- | --- |
| `web/umap.bin` | 128 KB (16384 × 2 × f32, exact) | `Float32Array(N, 2)` star positions for `/map` (only FE-fetched static asset) |
| `np-l20-res-16k/features/{i}.json.gz` | ~14.5 KB avg, ~230 MB total over 16384 files (range 5–220 KB) | per-feature dataset fetched from `GET https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{i}`, trimmed to: `index`, `explanations[].(description, explanationModelName, scoreV1, scoreV2, scores)`, `buckets[].(binMin, binMax, binContains, count, examples[].(tokens, values, max))`, `neighbors[].(idx, cos)` |

---


## 3. Dynamic data (Postgres DB)

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
  label            text not null check (char_length(label) between 1 and 1000),
  score            real check (score between 0 and 1),  -- null for seed rows
  scorer_model_id  text,                                -- null for seed rows; e.g. 'anthropic/claude-sonnet-4.5'
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

---

## 4. Bootstrap data flow

`python -m server.bootstrap` runs four idempotent steps. Required env: `DATABASE_URL`, `NEURONPEDIA_API_KEY`.

1. **`apply_schema`** — applies `server/schema.sql` to `DATABASE_URL` (creates `profiles`, `submissions`, view `feature_best`).
2. **`fetch_features`** — for `i ∈ [0, 16384)`, calls `GET https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{i}` with 8 concurrent workers and 5 retries (exponential backoff). Trims each response to the per-feature schema from §1 (`index`, `explanations[…]`, `buckets[…]`, `neighbors[…]`) and writes `np-l20-res-16k/features/{i}.json.gz`. Resumable: existing files are skipped.
3. **`build_umap_bin`** — downloads `google/gemma-scope-2b-pt-res :: layer_20/width_16k/average_l0_71/params.npz` from HuggingFace, loads `W_dec` `(16384, 2304)`, runs `umap.UMAP(metric="cosine")`, writes `web/umap.bin` as little-endian `Float32Array(16384, 2)` (~130 KB), overwriting any existing file. Deletes the downloaded `params.npz` (and its HuggingFace cache entry) once `web/umap.bin` is on disk — `W_dec` is not used at runtime. 
4. **`seed_submissions`** — reads `np-l20-res-16k/features/{i}.json.gz`. Ensures a `neuronpedia` profile exists (creates one with a random argon2 password if absent). Deletes any prior submissions for that user, then for every explanation with a non-empty `description` inserts `(user_id=neuronpedia, feature_id, label=description[:1000], score=NULL, scorer_model_id=NULL, created_at='epoch')` into `submissions`. The `'epoch'` (1970-01-01) sentinel marks rows as predating the game so any real user submission outranks them via the `score desc, created_at asc` ordering of `feature_best`. Idempotent per `neuronpedia` user (prior seed rows are wiped on each run).

---
