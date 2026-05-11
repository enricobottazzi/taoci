#!/usr/bin/env python3
"""Download the SAE decoder directions (W_dec) from HuggingFace.

Defaults match `source.jsonl` for Gemma-Scope layer 20, width 16k (L0=71).
Output: a single .npy of shape (d_sae, d_in) = (16384, 2304).
"""
import argparse
import numpy as np
from huggingface_hub import hf_hub_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--folder", default="layer_20/width_16k/average_l0_71")
    p.add_argument("--out", default="np-l20-res-16k/vectors.npy")
    a = p.parse_args()

    path = hf_hub_download(repo_id=a.repo, filename=f"{a.folder}/params.npz")
    W_dec = np.load(path)["W_dec"]
    np.save(a.out, W_dec)
    print(f"saved {W_dec.shape} {W_dec.dtype} -> {a.out}")


if __name__ == "__main__":
    main()
