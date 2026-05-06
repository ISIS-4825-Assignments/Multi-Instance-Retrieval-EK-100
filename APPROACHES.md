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
| 4 | AVION ViT-L + SMS Loss (10 ep, no inference flags) | **68.80** |
| 5 | AVION ViT-L + SMS Loss (10 ep, flip + clip-length 32) | **69.48** |
| 6 | AVION ViT-L + SMS Loss (16 ep, flip + clip-length 32) | **69.68** |

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

---

## Approach 4 — AVION ViT-L + SMS Loss

### Strategy

Fine-tunes the **AVION** video-language model (CLIP ViT-L backbone pre-trained on Ego4D via LaViLa) with **SMS Loss** (Symmetric Multi-Similarity Loss) on the EPIC-KITCHENS-100 retrieval training set. SMS Loss uses the ground-truth relevancy matrix to pull together positive pairs and push apart negatives in proportion to their relevance scores, giving a more nuanced supervision signal than binary contrastive losses.

**Step 1 — Base fine-tune** — the LaViLa ViT-L checkpoint is fine-tuned from scratch for 10 epochs using `ammplus_finetune.py` from the SMS-Loss repository:

```
torchrun --nproc_per_node=1 scripts/ammplus_finetune.py
  --model        CLIP_VITL14
  --batch-size   48
  --epochs       10
  --lr           2e-5
  --loss-margin  0.6
  --loss-thres   0.1
  --use-fast-conv1
  --grad-checkpointing
```

**Step 2 — Continue fine-tune (optional)** — training is resumed from the epoch-10 checkpoint for additional epochs using `--resume` and `--start-epoch 10`.

**Step 3 — Inference** — `test_mir.py` runs a forward pass over all test video segments and text queries to produce the similarity matrix. Two inference settings were compared:

| Setting | Flags |
|---|---|
| Standard | (none beyond required args) |
| TTA | `--flip --clip-length 32` |

`--flip` enables horizontal flip test-time augmentation (averages forward and flipped features). `--clip-length 32` increases the number of frames sampled per clip from the default (16) to 32, giving richer temporal context.

### Notebook

| Notebook | Purpose |
|---|---|
| `src/sms_avion/sms_avion.ipynb` | Installs dependencies, mounts Drive, downloads checkpoints, applies all source patches, fine-tunes the model, runs inference, and builds the Codabench submission ZIP |

### Data & Models

- **Backbone**: AVION LaViLa ViT-L pretrain checkpoint (`avion_pretrain_lavila_vitl_best.pt`, ~1.3 GB)
- **SMS-Loss repo**: `xqwang14/SMS-Loss` (cloned from GitHub)
- **Videos**: AVION pre-processed EK-100 clips (`EK100_320p_15sec_30fps_libx264`, 320p, 15 s chunks, 30 fps)
- **Annotations**: `EPIC_100_retrieval_train.csv`, `EPIC_100_retrieval_test.csv`, `caption_relevancy_EPIC_100_retrieval_train.pkl`

### Key Source Patches Applied

Several bugs in the SMS-Loss codebase required patching before training and inference could run:

| File | Patch |
|---|---|
| `scripts/ammplus_finetune.py` | Unpack `images, texts` in gradient-accumulation branch; replace undefined `args.accum_freq` with `args.update_freq`; safe fallbacks for missing `checkpoint['args']` fields; save a numbered `checkpoint_{epoch:04d}.pt` every epoch; disable live validation (not needed during training) |
| `avion/data/clip_dataset.py` | Retry `__getitem__` up to 20 times with a random index if `get_raw_item()` returns `None`; guard `relevancy_mat` load when `relevancy_test` path is absent |
| `scripts/test_mir.py` | Guard against missing `checkpoint['args']`; strip `module.` prefix from resumed state dicts; make `--relevancy-path` optional (saves submission first, then computes metrics only if the file exists); skip unused train-dataset construction |

### Files Generated

| File | Location | Description |
|---|---|---|
| `checkpoint_{epoch:04d}.pt` | `experiments/sms_vitl/` | Per-epoch full checkpoint (weights + optimizer + scaler) |
| `submission.pkl` | `experiments/sms_vitl/` | Raw similarity matrix with `vis_ids` and `txt_ids` |
| `submission.zip` | `experiments/sms_vitl/` | Codabench-compatible ZIP containing `test.pkl` (protocol-2 pickle, numpy compat fix applied) |

### Results

| Training epochs | Inference flags | nDCG AVG (Codabench) |
|---|---|---|
| 10 | — | **68.80** |
| 10 | `--flip --clip-length 32` | **69.48** |
| 16 | `--flip --clip-length 32` | **69.68** |

Test-time augmentation (`--flip`) and longer clip sampling (`--clip-length 32`) together add ~0.7 nDCG on top of the base inference. Extending fine-tuning from 10 to 16 epochs provides a further +0.2 nDCG improvement.
