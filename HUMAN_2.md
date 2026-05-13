# What's going on?

The Altas Of Unanchored Features [interface](https://enricobottazzi.github.io/taoci/) is an alternative experiment to the [main TAOCI experiment](https://taoci.ink/). The goal is always the same: to discover latent concepts. More specifically, a latent concept is a concept identified by a LLM that we don't have words for.

In this experiment we adopt a different approach: we try to **anchor** each feature against a fixed inventory of known human concepts, namely WordNet noun synsets. Those features that *don't* map cleanly onto any existing human concept are the candidates for latent undiscovered concepts.

## Pipeline

1. **Embed WordNet synsets.** Take every noun synset in WordNet (~82k) and embed its gloss (short definition) with `all-MiniLM-L6-v2`, producing one normalized vector per known concept. 

2. **Embed feature labels.** For each SAE feature in `np-l20-res-16k/features/*.json.gz`, pull its human-readable label, then extract an embedding.

3. **Compute nearest anchor.** Compute cosine similarity between every feature label and every synset gloss. For each feature, keep the single best-matching synset and its similarity score.

4. **Compute anchor score.** Convert each feature's best cosine into a percentile rank in [0, 1] across all features. Score ≈ 1 = label closely matches an existing WordNet concept (well-anchored). Score ≈ 0 = label is far from anything in WordNet (candidate latent concept).

## How to read the map

The map is a grid of stars. Each star is a 2D projection of a feature vector extracted from a LLM. 

Each star/feature is associated with: 
- a label, which a tentative description of the concept associated with that feature.
- the closest WordNet synset to that feature

At the LHS of the page you can see: 
- +/- buttons to zoom in and out the map.
- a slider to filter the stars by their anchor score.

Stars with low anchor score are candidates for latent undiscovered concepts.