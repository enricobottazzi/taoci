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

**Scoring procedure** — detection-style classification over `K=5` iterations:

1. **Build evidence set** from `np-l20-res-16k/features/{feature_id}.json.gz`:
   - `P` (positives): `top_activations(feature_id)` — the `TOP_K=5` deduped top-bucket examples (the same 5 that hydrate `/play`). Fixed across all iterations.
   - For the 5 neighbors of `feature_id` with highest `cos` (skipping self / out-of-range), take `top_activations(neighbor_id)` → 5 deduped distractors each → distractor pool `D` of 25.
   - Shuffle `D` with `Random(42)` and split into 5 disjoint contiguous subpools `D_1..D_5` of size 5.
2. **Run `K=5` iterations**. For each `i ∈ 1..5`:
   - `pool_i ← shuffle(P ∪ D_i, seed=42+i)` → 10 mixed-class examples.
   - Two LLM calls of 5 examples each: `call_A_i = pool_i[:5]`, `call_B_i = pool_i[5:]`. Each call uses the delphi detection prompt (system + 3 few-shots + user batch) via `OpenRouter("anthropic/claude-haiku-4.5", temperature=0)` and must return a Python list of 5 binary predictions.
   - Aggregate the 10 predictions of iteration `i` into `(tp_i, tn_i)` with `pos_i = neg_i = 5`.
3. **Reduce (pool-then-clip)** over all 50 predictions:
   `TP = Σ tp_i`, `TN = Σ tn_i`, `POS = NEG = 25`,
   `bal = 0.5·(TP/POS + TN/NEG)`, `score = max(0, 2·bal − 1) ∈ [0, 1]`.
   Per-iteration `bal_i` is logged for variance diagnostics but does not enter the final score.
4. **Persist** one row in `submissions` (`user_id`, `feature_id`, `label=description`, `score`, `scorer_model_id='anthropic/claude-haiku-4.5'`, `created_at=now()`). `is_new_best = score > max(prior score for this feature_id)`, computed over **all** prior rows (seeds + real). A real submission must clear the seed floor (`score > 0.5`) to be flagged as a new best on a feature that has no prior real submission.

Hard timeout 60 s on the full scoring run (all 10 calls) → 504. Any iteration where either call returns zero parseable selections is dropped from the pool; if fewer than 25 positive or 25 negative predictions remain, → 500.

---

Errors are uniform: `{ "error": "<code>", "message": "<human>" }`. 401 (auth), 400 (validation), 404 (unknown feature/user), 429 (rate-limit), 504 (scorer timeout), 500 (everything else).

---

## 2. Static data (read-only, immutable per release)

| path | size | notes |
| --- | --- | --- |
| `web/umap.bin` | 128 KB (16384 × 2 × f32, exact) | `Float32Array(N, 2)` star positions for `/map` (only FE-fetched static asset). On Fly: written to volume at `np-l20-res-16k/umap.bin`, symlinked into `web/` by bootstrap |
| `np-l20-res-16k/features/{i}.json.gz` | ~14.5 KB avg, ~230 MB total over 16384 files (range 5–220 KB) | per-feature dataset fetched from `GET https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{i}`, trimmed to: `index`, `explanations[].(description, explanationModelName, scoreV1, scoreV2, scores)`, `buckets[].(binMin, binMax, binContains, count, examples[].(tokens, values, max))`, `neighbors[].(idx, cos)` |

---


## 3. Dynamic data (Postgres DB)

Two tables (`profiles`, `submissions`) + one view (`feature_best`). Canonical DDL lives in [`server/schema.sql`](server/schema.sql) and is applied verbatim by `bootstrap.apply_schema`.

Notes:
- Passwords are hashed with argon2id (`passlib[argon2]`) and stored on `profiles`; JWTs are issued and verified by the server itself.
- Seed rows (see §4) carry `score=0.5` (policy floor — not a measured score) and `scorer_model_id=NULL`. Real `/score` writes carry a measured `score ∈ [0,1]` and a non-null `scorer_model_id`. The presence of `scorer_model_id` is the seed-vs-real discriminator.
- `feature_best` orders by `(score desc nulls last, created_at asc, submission_id asc)`. With `seed.created_at = epoch`, a real submission must achieve `score > 0.5` to displace a seed; ties on `0.5` go to the seed. `submission_id` is the final tie-breaker (bigserial → unique → deterministic; resolves ties among multiple seeds for the same feature in Neuronpedia's `explanations[]` insertion order).

---

## 4. Bootstrap data flow

`python -m server.bootstrap` runs four idempotent steps. Required env: `DATABASE_URL`, `NEURONPEDIA_API_KEY`.

1. **`apply_schema`** — applies `server/schema.sql` to `DATABASE_URL` (creates `profiles`, `submissions`, view `feature_best`).
2. **`fetch_features`** — for `i ∈ [0, 16384)`, calls `GET https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{i}` with 8 concurrent workers and 5 retries (exponential backoff). Trims each response to the per-feature schema from §1 (`index`, `explanations[…]`, `buckets[…]`, `neighbors[…]`) and writes `np-l20-res-16k/features/{i}.json.gz`. Resumable: existing files are skipped.
3. **`build_umap_bin`** — if `np-l20-res-16k/umap.bin` is absent, downloads `google/gemma-scope-2b-pt-res :: layer_20/width_16k/average_l0_71/params.npz` from HuggingFace, loads `W_dec` `(16384, 2304)`, runs `umap.UMAP(metric="cosine")`, writes the embedding as little-endian `Float32Array(16384, 2)` (~130 KB) to the volume. Always (re)creates `web/umap.bin` as a symlink to the volume copy so `StaticFiles` can serve it. `W_dec` is not used at runtime.
4. **`seed_submissions`** — reads `np-l20-res-16k/features/{i}.json.gz`. Ensures a `neuronpedia` profile exists (creates one with a random argon2 password if absent). If that user already has any submission rows, skips (idempotent fast path). Otherwise inserts, for every explanation with a non-empty `description`, `(user_id=neuronpedia, feature_id, label=description.strip()[:1000], score=0.5, scorer_model_id=NULL, created_at='epoch')` into `submissions`. `score=0.5` is a policy floor (not a measured value); it forces real submissions to clear a non-trivial informedness bar before displacing the seed on `feature_best`. `'epoch'` (1970-01-01) is a sentinel marking origin and is not load-bearing for ordering. To re-seed, manually `delete from submissions where user_id = (select id from profiles where username='neuronpedia')` and rerun.

---
