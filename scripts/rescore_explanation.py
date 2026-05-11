#!/usr/bin/env python3
"""Re-run delphi's DetectionScorer (== eleuther_recall) for one feature.

Loads the chosen explanation, activating examples and hard-negative
distractors (top-activating contexts of the feature's nearest neighbours)
from the local S3 dump, then hands them to delphi's pipeline with any
OpenRouter model as the scorer LLM. Prints balanced accuracy.
"""
import argparse, asyncio, gzip, json, os, random, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from delphi.clients import OpenRouter
from delphi.latents import (ActivatingExample, Latent, LatentRecord,
                            NonActivatingExample)
from delphi.scorers.classifier.detection import DetectionScorer

ROOT = Path("np-l20-res-16k")
MODULE = "blocks.20.hook_resid_post"


def rows(folder: str, idx: int) -> list[dict]:
    with gzip.open(ROOT / folder / f"batch-{idx // 1024}.jsonl.gz", "rt") as f:
        return [d for line in f if (d := json.loads(line))["index"] == str(idx)]


def to_example(row: dict, tok, cls):
    ids = tok.convert_tokens_to_ids(row["tokens"])
    return cls(tokens=torch.tensor(ids), activations=torch.tensor(row["values"]),
               str_tokens=row["tokens"])


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--explainer", required=True, help="e.g. gemini-2.5-flash-lite")
    p.add_argument("--scorer-model", default="anthropic/claude-haiku-4.5")
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--n-distractors", type=int, default=20)
    p.add_argument("--n-shown", type=int, default=5)
    p.add_argument("--tokenizer", default="unsloth/gemma-2-2b",
                   help="ungated mirror of google/gemma-2-2b (same vocab)")
    a = p.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    exp = next((e for r in rows("explanation-scores", a.feature)
                for e in r["explanations"] if e["explanationModelName"] == a.explainer),
               None) or sys.exit(f"no explanation by '{a.explainer}' for feature {a.feature}")
    description = exp["description"]
    print(f"description: {description!r}")
    print("existing scores:")
    for s in exp.get("scores") or []:
        print(f"  {s['explanationScoreTypeName']:<18s} "
              f"by {s['explanationScoreModelName']:<22s} = {s['value']}")
    if not exp.get("scores"):
        print("  (none)")

    test = [to_example(r, tok, ActivatingExample)
            for r in rows("activations", a.feature)][:a.n_test]

    nbrs = [i for i in rows("features", a.feature)[0]["topkCosSimIndices"]
            if i != a.feature]
    pool: list[dict] = []
    for n in nbrs:
        pool.extend(rows("activations", n))
        if len(pool) >= a.n_distractors * 3:
            break
    random.Random(42).shuffle(pool)
    not_active = [to_example(r, tok, NonActivatingExample)
                  for r in pool[:a.n_distractors]]

    record = LatentRecord(latent=Latent(MODULE, a.feature), test=test,
                          not_active=not_active, explanation=description)
    scorer = DetectionScorer(client=OpenRouter(a.scorer_model, api_key=key),
                             n_examples_shown=a.n_shown, verbose=False)
    outs = (await scorer(record)).score
    pos = sum(o.activating for o in outs)
    neg = len(outs) - pos
    tp = sum(o.correct for o in outs if o.activating)
    tn = sum(o.correct for o in outs if not o.activating)
    bal = 0.5 * (tp / max(pos, 1) + tn / max(neg, 1))
    print(f"balanced accuracy = {bal:.3f}   (TPR={tp}/{pos}, TNR={tn}/{neg})")


if __name__ == "__main__":
    asyncio.run(main())
