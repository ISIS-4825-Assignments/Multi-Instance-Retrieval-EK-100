# EK-100 Multi-Instance Retrieval — Approaches

Competition: [Codabench EK-100 MIR](https://www.codabench.org/competitions/12008)  
Task: Given a text query, rank 9 668 video segments (and vice versa) by relevance.  
Primary metric: **nDCG AVG** (average of video→text and text→video nDCG).

---

## Iteration Summary

| # | Approach | nDCG AVG |
|---|---|---|
| 1 | JPoSE Base | **53.53** |
| 2 | JPoSE Ensemble (no re-ranking) | **55.31** |
| 3 | JPoSE Ensemble + Re-ranking | **55.82** |

---

## Approach 1 — JPoSE Base

### Strategy

Runs inference with the pre-trained **JPoSE** (Joint Part-of-Speech Embeddings) model. JPoSE embeds video clips and text queries into a shared space decomposed by grammatical role: verb, noun, and action. At test time the three sub-embeddings are concatenated (`comb-func = cat`) to produce a joint representation, and cosine similarity between all video–text pairs is computed to produce the final similarity matrix.

No training is done. Only the best released checkpoint (`JPoSE_BEST`) is used.

### Notebooks

| Notebook | Purpose |
|---|---|
| `src/jpose_base/jpose_base_data.ipynb` | Downloads `JPoSE_data.zip` (~1.64 GB) and the EPIC-100 retrieval annotations to Google Drive |
| `src/jpose_base/jpose_base.ipynb` | Extracts data, patches PyTorch compatibility, runs JPoSE inference, and packages the submission ZIP |

### Data & Models

- **JPoSE repo**: `Joint-Part-of-Speech-Embeddings` (cloned from GitHub)
- **Checkpoint**: `JPoSE_BEST/model/EPIC_100_retrieval_JPoSE_BEST.pth`
- **Features**: pre-extracted video features and text features included in `JPoSE_data.zip`

### Files Generated

| File | Location | Description |
|---|---|---|
| `JPoSE_BEST_test_latest.pkl` | `EK100_MIR/submissions/` | Raw similarity matrix (9668 × 3842, float32) with `vis_ids` and `txt_ids` |
| `JPoSE_BEST_submission.zip` | `EK100_MIR/submission_zips/` | Codabench-compatible ZIP containing `test.pkl` |

### Result

| Metric | VT | TV | AVG |
|---|---|---|---|
| nDCG | 0.707 | 0.674 | **0.690** (train-set val) |
| **Codabench nDCG** | — | — | **53.53** |

---

## Approach 2 — JPoSE Ensemble

### Strategy

Combines three independent retrieval models by averaging their normalized similarity matrices, with an optional diffusion-based re-ranking step applied on top.

**Step 1 — Independent inference** — three models are run separately, each producing a 9668 × 3842 similarity matrix:

| Model | Architecture | Features |
|---|---|---|
| **JPoSE** | Part-of-Speech joint embedding, triplet loss | Pre-extracted video + text features |
| **MI-MM** | Multi-Instance Multi-Modal matching | S3D HowTo100M video features |
| **MLP / MMEN** | Multi-Modal Embedding Network, caption-based | Pre-extracted video + text features |

**Step 2 — Ensemble** — each matrix is min-max normalized row-wise to [0, 1], then averaged with equal weights (1/3 each):

```
sim_ensemble = (1/3) * norm(sim_mimm)
             + (1/3) * norm(sim_jpose)
             + (1/3) * norm(sim_mlp)
```

**Step 3 — Re-ranking (optional)** — a k-NN affinity diffusion step propagates scores through visual-visual and text-text neighborhood graphs:

- For each query row, the top-k most similar videos form a visual affinity graph (`A_vv`).
- For each text query, the same is done in text space (`A_tt`).
- Both affinities are applied to the ensemble matrix and blended with the original scores:

```
sim_reranked = (1 - alpha) * sim_ensemble + alpha * 0.5 * (A_vv @ S + (A_tt @ S.T).T)
```

Default parameters: `k = 20`, `alpha = 0.3`.

### Notebooks

| Notebook | Purpose |
|---|---|
| `src/jpose_ensemble/jpose_ensemble_data.ipynb` | Downloads `JPoSE_data.zip` (~1.64 GB), `MI-MM_data.zip` (~0.68 GB), and annotations to Google Drive |
| `src/jpose_ensemble/jpose_ensemble.ipynb` | Extracts both datasets, patches compatibility issues, runs all three model inferences, computes ensemble and re-ranked matrices, and packages all submission ZIPs |

### Data & Models

- **JPoSE repo**: `Joint-Part-of-Speech-Embeddings` (cloned from GitHub)
- **MI-MM repo**: `MI-MM` (cloned from GitHub)
- **JPoSE checkpoint**: `JPoSE_BEST/model/EPIC_100_retrieval_JPoSE_BEST.pth` (~8 MB)
- **MLP checkpoint**: `MMEN_BEST/model/EPIC_100_retrieval_MLP_BEST.pth` (~4 MB)
- **MI-MM checkpoint**: `MI-MM/data/models/` (~154 MB, best epoch 202)
- **S3D features**: `MI-MM/data/features/` (~315 MB)

### Files Generated

**Intermediate similarity matrices** (saved to `EK100_MIR/submissions/`):

| File | Model |
|---|---|
| `MI-MM_test_latest.pkl` | MI-MM |
| `JPoSE_BEST_test_latest.pkl` | JPoSE |
| `MMEN_BEST_test_latest.pkl` | MLP / MMEN |

**Submission ZIPs** (saved to `EK100_MIR/submission_zips/`):

| File | Contents |
|---|---|
| `ensemble_submission.zip` | Ensemble (no re-ranking) |
| `reranked_submission.zip` | Ensemble + re-ranking |
| `JPoSE_submission.zip` | JPoSE alone |
| `MI-MM_submission.zip` | MI-MM alone |
| `MLP_submission.zip` | MLP / MMEN alone |

### Results

| Variant | nDCG AVG (Codabench) |
|---|---|
| Ensemble only | **55.31** |
| Ensemble + re-ranking (k=20, α=0.3) | **55.82** |
