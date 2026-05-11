#!/usr/bin/env python3
"""Re-run delphi's DetectionScorer (== eleuther_recall) for one feature.

Loads activating examples and hard-negative distractors (top-activating
contexts of the feature's nearest neighbours) from the local S3 dump and
hands them to delphi's pipeline with any OpenRouter model as the scorer LLM.
The explanation is either fetched from `explanation-scores` (`--explainer`)
or provided verbatim (`--description`). Prints balanced accuracy.
"""
import argparse, asyncio, gzip, json, os, random, sys
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

from delphi.clients import OpenRouter
from delphi.latents import (ActivatingExample, Latent, LatentRecord,
                            NonActivatingExample)
from delphi.scorers.classifier.detection import DetectionScorer

ROOT = Path(__file__).resolve().parent.parent / "np-l20-res-16k"
MODULE = "blocks.20.hook_resid_post"
DEFAULT_TOKENIZER = "unsloth/gemma-2-2b"


def rows(folder: str, idx: int) -> list[dict]:
    with gzip.open(ROOT / folder / f"batch-{idx // 1024}.jsonl.gz", "rt") as f:
        return [d for line in f if (d := json.loads(line))["index"] == str(idx)]


def to_example(row: dict, tok, cls):
    ids = tok.convert_tokens_to_ids(row["tokens"])
    return cls(tokens=torch.tensor(ids), activations=torch.tensor(row["values"]),
               str_tokens=row["tokens"])


@lru_cache(maxsize=2)
def get_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name)


def lookup_description(feature: int, explainer: str) -> str:
    exp = next((e for r in rows("explanation-scores", feature)
                for e in r["explanations"]
                if e["explanationModelName"] == explainer), None)
    if not exp:
        raise ValueError(f"no explanation by {explainer!r} for feature {feature}")
    return exp["description"]


async def score_explanation(
    feature: int,
    description: str,
    scorer_model: str = "anthropic/claude-sonnet-4.5",
    n_test: int = 20,
    n_distractors: int = 20,
    n_shown: int = 5,
    tokenizer: str = DEFAULT_TOKENIZER,
    api_key: str | None = None,
) -> dict:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    tok = get_tokenizer(tokenizer)

    test = [to_example(r, tok, ActivatingExample)
            for r in rows("activations", feature)][:n_test]

    nbrs = [i for i in rows("features", feature)[0]["topkCosSimIndices"]
            if i != feature]
    pool: list[dict] = []
    for n in nbrs:
        pool.extend(rows("activations", n))
        if len(pool) >= n_distractors * 3:
            break
    random.Random(42).shuffle(pool)
    not_active = [to_example(r, tok, NonActivatingExample)
                  for r in pool[:n_distractors]]

    record = LatentRecord(latent=Latent(MODULE, feature), test=test,
                          not_active=not_active, explanation=description)
    scorer = DetectionScorer(client=OpenRouter(scorer_model, api_key=key),
                             n_examples_shown=n_shown, verbose=False)
    outs = (await scorer(record)).score or []
    if not outs:
        raise RuntimeError(f"scorer returned no parseable selections "
                           f"(model={scorer_model}); try a stronger model")
    pos = sum(o.activating for o in outs)
    neg = len(outs) - pos
    tp = sum(o.correct for o in outs if o.activating)
    tn = sum(o.correct for o in outs if not o.activating)
    bal = 0.5 * (tp / max(pos, 1) + tn / max(neg, 1))
    return {"balanced_accuracy": bal, "tp": tp, "tn": tn, "pos": pos, "neg": neg}


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--explainer", help="lookup explanation by model name")
    g.add_argument("--description", help="use this explanation verbatim")
    p.add_argument("--scorer-model", default="anthropic/claude-sonnet-4.5")
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--n-distractors", type=int, default=20)
    p.add_argument("--n-shown", type=int, default=5)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    a = p.parse_args()

    description = a.description or lookup_description(a.feature, a.explainer)
    print(f"description: {description!r}")
    r = await score_explanation(a.feature, description, a.scorer_model,
                                a.n_test, a.n_distractors, a.n_shown, a.tokenizer)
    print(f"balanced accuracy = {r['balanced_accuracy']:.3f}   "
          f"(TPR={r['tp']}/{r['pos']}, TNR={r['tn']}/{r['neg']})")


if __name__ == "__main__":
    asyncio.run(main())
