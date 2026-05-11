#!/usr/bin/env python3
"""UMAP-project SAE decoder directions and render a 2D scatter.

Input:  np-l20-res-16k/vectors.npy            (16384, 2304) unit-norm
Output: np-l20-res-16k/umap.npy               (16384, 2) embedding (cached)
        np-l20-res-16k/umap.png               static scatter
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", default="np-l20-res-16k/vectors.npy")
    p.add_argument("--out-embed", default="np-l20-res-16k/umap.npy")
    p.add_argument("--out-png", default="np-l20-res-16k/umap.png")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recompute", action="store_true")
    a = p.parse_args()

    embed_path = Path(a.out_embed)
    if embed_path.exists() and not a.recompute:
        emb = np.load(embed_path)
        print(f"loaded cached {emb.shape} <- {embed_path}")
    else:
        import umap  # lazy: heavy import
        W = np.load(a.vectors)
        print(f"fitting UMAP on {W.shape} (metric={a.metric}, "
              f"n_neighbors={a.n_neighbors}, min_dist={a.min_dist})")
        emb = umap.UMAP(n_neighbors=a.n_neighbors, min_dist=a.min_dist,
                        metric=a.metric, random_state=a.seed).fit_transform(W)
        np.save(embed_path, emb.astype(np.float32))
        print(f"saved {emb.shape} -> {embed_path}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(emb[:, 0], emb[:, 1], s=1, alpha=0.3, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UMAP of W_dec ({len(emb)} features)")
    fig.tight_layout(); fig.savefig(a.out_png)
    print(f"wrote {a.out_png}")


if __name__ == "__main__":
    main()
