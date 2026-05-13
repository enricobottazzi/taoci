import base64, gzip, json
from pathlib import Path
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
import numpy as np

LABEL_MAX = 1000
CACHE = Path('synset_cache.npz')
LABEL_CACHE = Path('label_cache.npz')

model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Embed synset glosses (cached)
if CACHE.exists():
    cache = np.load(CACHE, allow_pickle=True)
    synset_vecs, synset_ids, synset_glosses = cache['vecs'], list(cache['ids']), list(cache['glosses'])
else:
    synsets = list(wn.all_synsets(pos='n'))
    synset_glosses = [s.definition() for s in synsets]
    synset_ids = [s.name() for s in synsets]
    synset_vecs = model.encode(synset_glosses, normalize_embeddings=True, batch_size=128)
    np.savez(CACHE, vecs=synset_vecs, ids=synset_ids, glosses=synset_glosses)

# 2. Load feature labels and score
def first_label(rec):
    for e in rec.get('explanations') or []:
        label = (e.get('description') or '').strip()[:LABEL_MAX]
        if label:
            return label
    return None

if LABEL_CACHE.exists():
    lc = np.load(LABEL_CACHE, allow_pickle=True)
    feature_ids, labels, label_vecs = lc['feature_ids'], list(lc['labels']), lc['vecs']
else:
    rows = []
    for path in sorted(Path('np-l20-res-16k/features').glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            rec = json.load(f)
        label = first_label(rec)
        if label:
            rows.append((int(rec['index']), label))
    feature_ids, labels = zip(*rows)
    label_vecs = model.encode(list(labels), normalize_embeddings=True, batch_size=128)
    np.savez(LABEL_CACHE, feature_ids=np.array(feature_ids), labels=np.array(labels), vecs=label_vecs)

sims = label_vecs @ synset_vecs.T
best_synset = np.argmax(sims, axis=1)
best_cos = sims[np.arange(len(sims)), best_synset]

# 3. Score each feature in [0, 1] via percentile rank
anchor_score = best_cos.argsort().argsort() / (len(best_cos) - 1)

features = {
    int(fid): {
        'label':  labels[i],
        'anchor': synset_ids[best_synset[i]],
        'gloss':  synset_glosses[best_synset[i]],
        'score':  float(anchor_score[i]),
    }
    for i, fid in enumerate(feature_ids)
}

# 4. Render the anchor map (offline HTML)
payload = {
    'positions_b64': base64.b64encode(Path('web/umap.bin').read_bytes()).decode(),
    'features': features,
}
template = Path('map_wordnet.html').read_text()
html = template.replace('/*__DATA__*/', f'window.__DATA__ = {json.dumps(payload)};')
Path('anchor_map.html').write_text(html)
