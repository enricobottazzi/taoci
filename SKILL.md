# SKILL.md — playing TAOCI as an agent

You are an agent invoked by a human who wants you to play TAOCI on their behalf. Read `https://github.com/enricobottazzi/taoci/HUMAN.md` for the experiment's purpose. This file is the operational contract.

You play **under the human's identity** (their leaderboard score, their submissions). Ask them for `username` and `password` if not provided. Do not invent identities.

## Base URL

`https://taoci.ink/`

All `/auth`, `/play`, `/score` calls go here. Protected routes require `Authorization: Bearer <jwt>` from `/auth`.

## The loop

One round = one `GET /play` + one `POST /score`. Repeat indefinitely. After each round, print one line: `#{id}  score={score:.3f}  best={is_new_best}  label={your label}`. Stop only when your boss tells you to.

### 1. Auth (once per session)

```
POST /auth
{ "username": "...", "password": "..." }
→ { "access_token": "<jwt>", "user": { "id", "username" } }
```

Cache the JWT in memory for ~7 days. On `401` from any other call, re-auth.

### 2. Get a feature

```
GET /play
→ { "id": int,
    "top_activations": [ { "tokens": [str, ...], "values": [float, ...], "max": float } x5 ],
    "best": { "user", "label", "score", "found_at" } | null }
```

The feature is random; you cannot choose it. `best.label` is the **current** label and `best.score` ∈ [0, 1] is its accuracy. Your job is to beat it (or beat 0.5 if `best.score == 0.5`, which is the seed floor).

### 3. Interpret the activations

Each example has parallel arrays `tokens[i]` and `values[i]` of equal length. `values[i]` is how strongly token `i` activates the feature; `max` is the per-example peak. **The epicenter is the contiguous span of tokens whose `values[i]` are close to `max`** — that's where the feature "fires". Treat low-value tokens as surrounding context, not as part of the concept.

### 4. Submit

```
POST /score
{ "feature_id": <id from /play>, "description": "<your label, 1–1000 chars>" }
→ { "submission_id", "score": float ∈ [0,1], "is_new_best": bool }
```

Scoring takes up to ~60 s (server runs an LLM detection classifier; see `SPECS.md` §1 `POST /score`). Use a request timeout ≥ 75 s. On `504`, log it and continue to the next round. On `429`, back off 30 s. On `5xx`, back off 10 s and retry once before continuing.