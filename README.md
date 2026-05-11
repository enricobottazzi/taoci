# taoci

## 1. Download the dataset

```bash
# (a) Activations, explanations, features metadata
aws s3 cp --no-sign-request --recursive \
  s3://neuronpedia-datasets/v1/gemma-2-2b/20-gemmascope-res-16k/ \
  ./np-l20-res-16k/

# (b) SAE decoder directions W_dec -> np-l20-res-16k/vectors.npy  (16384, 2304)
pip install numpy huggingface_hub
python scripts/fetch_vectors.py

# (c) Per-(feature, explanation) scores -> np-l20-res-16k/explanation-scores/batch-*.jsonl.gz
# Not in the S3 dump; fetched via Neuronpedia API (~16k requests, resumable).
export NEURONPEDIA_API_KEY=...   # https://neuronpedia.org/account
python scripts/fetch_scores.py
```

## 2. Explore dataset (a)

Top-level metadata (one JSON object each, pretty-printed):

```bash
for f in np-l20-res-16k/*.jsonl; do echo "=== $f ==="; jq . "$f"; done
```

Per-feature data is split across three subfolders, each batched as `batch-*.jsonl.gz` (1024 features/file → 16 batches × 16384 features). The `index` field is the join key. We'll walk feature `877` (batch `0 = 877/1024`) through all three.

```bash
IDX=877; B=$((IDX/1024))
```

### Features (`features/`) — static metadata, **1 row = 1 feature**

```bash
gzcat np-l20-res-16k/features/batch-$B.jsonl.gz | head -1 | jq 'keys'
```

Notice that the `vector` associated to a feature is empty here. We will fetch it later.

### Activations (`activations/`) — top-k max-activating examples, **1 row = 1 (feature, example)**

```bash
gzcat np-l20-res-16k/activations/batch-$B.jsonl.gz | head -1 | jq 'keys'
```

### Explanations (`explanations/`) — LLM-generated descriptions, **1 row = 1 (feature, explainer)**

```bash
gzcat np-l20-res-16k/explanations/batch-$B.jsonl.gz | head -1 | jq 'keys'
gzcat np-l20-res-16k/explanations/batch-$B.jsonl.gz \
  | jq -c "select(.index==\"$IDX\") | {description,explanationModelName,typeName}"
```

## 3. Explore dataset (b)

`vectors.npy` is the SAE decoder matrix `W_dec`: row `i` is the unit-norm direction in residual-stream space that feature `i` writes into.

```bash
python - <<'PY'
import numpy as np
W = np.load("np-l20-res-16k/vectors.npy")
print(W.shape, W.dtype)                       # (16384, 2304) float32
print("norms:", np.linalg.norm(W, axis=1)[:5])  # all == 1.0 (unit-norm)

i = 877
sim = W @ W[i]                                # cosine sim (rows already unit-norm)
top = np.argsort(-sim)[:6]
print("nearest to 877:", list(zip(top.tolist(), sim[top].round(3).tolist())))
PY
```

Cross-check against the precomputed `topkCosSim*` fields in `features/`:

```bash
gzcat np-l20-res-16k/features/batch-0.jsonl.gz \
  | jq -c 'select(.index=="877") | {topkCosSimIndices,topkCosSimValues}'
```

## 4. Explore dataset (c)

`explanation-scores/batch-*.jsonl.gz` mirrors the 16-batch layout from (a). **1 row = 1 feature**, holding *all* its explanations (across explainer models) and each explanation's scorer outputs.

```bash
gzcat np-l20-res-16k/explanation-scores/batch-$B.jsonl.gz \
  | jq -c "select(.index==\"$IDX\") | {index, n_exps: (.explanations|length), explainers:[.explanations[].explanationModelName]}"
```

Drill into one explanation + its scores (e.g. feature 0):

```bash
gzcat np-l20-res-16k/explanation-scores/batch-0.jsonl.gz \
  | jq -c 'select(.index=="0") | .explanations[] | {description, explanationModelName,
      scores: [.scores[]? | {explanationScoreTypeName, explanationScoreModelName, value}]}'
```

Coverage is sparse: ~50k explanations across 16k features, but only ~1k carry scorer outputs (the rest were generated but not yet auto-scored). Each `scores[*]` entry records *which scorer model* (e.g. `claude-4-5-haiku`) ran *which scorer type* (e.g. `eleuther_fuzz`, `eleuther_detection`) and the resulting `value ∈ [0,1]`. Detailed per-example outputs live in `jsonDetails`.

## 5. UMAP of W_dec

```bash
pip install umap-learn matplotlib
python scripts/umap_vectors.py     # caches np-l20-res-16k/umap.npy, writes umap.png
# tweak: --n-neighbors 30 --min-dist 0.0 --metric cosine --recompute
```

Cosine metric matches the (already unit-norm) row geometry. The embedding is cached; re-run with `--recompute` to refit.

## TODO

- [x] UMAP
- [ ] Clustering (Hexbin techniques + colouring) via hexagons (viz techniques)

