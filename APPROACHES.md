# EK-100 Multi-Instance Retrieval — Approaches

Competition: [Codabench EK-100 MIR](https://www.codabench.org/competitions/12008)
Task: Given a text query, rank 9,668 video segments (and vice versa) by relevance.
Primary metric: **nDCG AVG** (average of video→text and text→video nDCG).

This document is the technical companion to the [README](README.md). The README explains *how to run* every notebook; this document explains *what each iteration does, why, and with what hyperparameters and patches*.

---

## **Iteration Summary**

| # | Approach | nDCG AVG (Codabench) |
|---|---|---|
| 1 | JPoSE Base | **53.53** |
| 2 | JPoSE Ensemble (no re-ranking) | **55.31** |
| 3 | JPoSE Ensemble + Re-ranking | **55.82** |
| 4 | AVION ViT-L + SMS Loss (10 ep, standard inference) | **68.80** |
| 5 | AVION ViT-L + SMS Loss (10 ep, flip + clip-length 32) | **69.48** |
| 6 | AVION ViT-L + SMS Loss (10 + 6 ep, flip + clip-length 32) | **69.68** |

Iteration 1 corresponds to **Approach 1**. Iterations 2–3 correspond to **Approach 2**. Iterations 4–6 correspond to **Approach 3**.

---

## **Approach 1 — JPoSE Base**

*Covers iteration 1.*

### Strategy

Runs inference with the pre-trained **JPoSE** (Joint Part-of-Speech Embeddings) model on the EK-100 retrieval test set. JPoSE embeds video clips and text queries into a shared space decomposed by grammatical role: verb, noun, and action. At test time, the three sub-embeddings are concatenated (`comb-func = cat`) to form a joint representation, and cosine similarity between all video–text pairs produces the final 9668 × 3842 similarity matrix.

No fine-tuning is performed. Only the best released checkpoint (`JPoSE_BEST`) is used.

### Notebooks (run in order)

| Notebook | Sections | Purpose |
|---|---|---|
| `src/jpose_base/jpose_base_data.ipynb` | 1–2 | Downloads `JPoSE_data.zip` (~1.64 GB) and EPIC-100 retrieval annotation files to Google Drive. |
| `src/jpose_base/jpose_base.ipynb` | 1–9 | GPU check → Drive mount → dep install → clone JPoSE → paths → extract → compatibility patches → inference → submission ZIP. |

### Data & Models

| Item | Size | Source |
|---|---|---|
| `JPoSE_data.zip` | ~1.64 GB | Downloaded by `jpose_base_data.ipynb` from Dropbox |
| `JPoSE_BEST` checkpoint | ~8 MB | Included in `JPoSE_data.zip` |
| Pre-extracted video features | ~1,955 MB | Included in `JPoSE_data.zip` |
| Pre-extracted text features | ~107 MB | Included in `JPoSE_data.zip` |
| EPIC-100 annotation PKLs | ~12 MB | Downloaded by `jpose_base_data.ipynb` from EPIC-KITCHENS GitHub |

All data is stored under `MyDrive/EK100_MIR/data/` on Google Drive.

### Compatibility Patches Applied

| File | Patch |
|---|---|
| All `*.py` in `Joint-Part-of-Speech-Embeddings/src/` | Regex-replace all `torch.load(…)` calls → `torch.load(…, weights_only=False)` to fix the PyTorch ≥ 2.0 `weights_only` default change. |

### Files Generated

| File | Location | Description |
|---|---|---|
| `JPoSE_BEST_test_latest.pkl` | `EK100_MIR/submissions/` | Raw similarity matrix (9668 × 3842, float32) with `vis_ids` and `txt_ids`. |
| `JPoSE_BEST_submission.zip` | `EK100_MIR/submission_zips/` | Codabench-compatible ZIP containing `test.pkl` (protocol-2 pickle, numpy compat patch applied). |

### Results

| Metric | VT | TV | AVG |
|---|---|---|---|
| nDCG (train-set val) | 0.707 | 0.674 | **0.690** |
| mAP (train-set val) | 0.757 | 0.712 | **0.734** |
| **nDCG AVG (Codabench)** | — | — | **53.53** |

---

## **Approach 2 — JPoSE Ensemble + Re-ranking**

*Covers iterations 2 and 3.*

### Strategy

Combines three independent retrieval models into a single similarity matrix via score-level fusion, then optionally applies a graph-diffusion re-ranking step.

**Step 1 — Independent inference.** Three models are run separately, each producing a 9668 × 3842 similarity matrix:

| Model | Architecture | Training | Features |
|---|---|---|---|
| **JPoSE** | Part-of-Speech joint embedding, triplet loss | Pre-trained; no fine-tuning | Pre-extracted video + text features |
| **MI-MM** | Multi-Instance Multi-Modal matching | Pre-trained; no fine-tuning | S3D HowTo100M video features |
| **MLP / MMEN** | Multi-Modal Embedding Network, caption-based | Pre-trained; no fine-tuning | Pre-extracted video + text features |

**Step 2 — Ensemble.** Each matrix is min-max normalized row-wise to [0, 1] (so models with different score scales contribute equally), then averaged with equal weights:

```
sim_ensemble = (1/3) × normalize(sim_mimm)
             + (1/3) × normalize(sim_jpose)
             + (1/3) × normalize(sim_mlp)
```

Weights can be tuned on a validation set if one is available; equal weights were used here.

**Step 3 — Re-ranking (optional).** A k-NN affinity diffusion step propagates scores through visual-visual and text-text neighbourhood graphs:

1. Each query's similarity row is L2-normalized; the top-k most similar videos form a sparse visual affinity matrix `A_vv`.
2. The same is done transposed in text space to form `A_tt`.
3. Expanded scores are computed as `S_exp = 0.5 × (A_vv @ S + (A_tt @ Sᵀ)ᵀ)`.
4. The final re-ranked matrix blends the original and expanded scores:

```
sim_reranked = (1 − α) × sim_ensemble + α × S_exp
```

Default parameters: `k = 20`, `α = 0.3`.

### Notebooks (run in order)

| Notebook | Sections | Purpose |
|---|---|---|
| `src/jpose_ensemble/jpose_ensemble_data.ipynb` | 1–2 | Downloads `JPoSE_data.zip` (~1.64 GB), `MI-MM_data.zip` (~0.68 GB), and EPIC-100 annotation files to Google Drive. |
| `src/jpose_ensemble/jpose_ensemble.ipynb` | 1–9 | GPU check → Drive mount → dep install → clone JPoSE & MI-MM → paths → extract → compatibility patches → tri-model inference + ensemble + re-rank → five submission ZIPs. |

### Data & Models

| Item | Size | Source |
|---|---|---|
| `JPoSE_data.zip` | ~1.64 GB | Downloaded by `jpose_ensemble_data.ipynb` from Dropbox |
| `MI-MM_data.zip` | ~0.68 GB | Downloaded by `jpose_ensemble_data.ipynb` from Dropbox |
| `JPoSE_BEST` checkpoint | ~8 MB | Included in `JPoSE_data.zip` |
| `MMEN_BEST` (MLP) checkpoint | ~4 MB | Included in `JPoSE_data.zip` |
| MI-MM checkpoint (best epoch 202) | ~154 MB | Included in `MI-MM_data.zip` |
| S3D HowTo100M features (MI-MM) | ~315 MB | Included in `MI-MM_data.zip` |
| Pre-extracted JPoSE video features | ~1,955 MB | Included in `JPoSE_data.zip` |
| Pre-extracted JPoSE text features | ~107 MB | Included in `JPoSE_data.zip` |

### Compatibility Patches Applied

| File | Patch |
|---|---|
| `MI-MM/src/loader/loader_features.py` | Replace `import pickle5 as pickle` → `import pickle` (pickle5 is built into Python 3.8+ stdlib). |
| `MI-MM/src/testing.py` | Add `weights_only=False` to `th.load(…)` for PyTorch ≥ 2.0 compatibility. |
| `MI-MM/src/models/embedding_projection.py` | Add `weights_only=False` to `th.load(…)` for the S3D pretrain weights. |
| All `*.py` in `Joint-Part-of-Speech-Embeddings/src/` | Regex-replace all `torch.load(…)` → `torch.load(…, weights_only=False)`. |

### Files Generated

**Intermediate submission pickles** (saved to `EK100_MIR/submissions/`):

| File | Model |
|---|---|
| `MI-MM_test_latest.pkl` | MI-MM |
| `JPoSE_BEST_test_latest.pkl` | JPoSE |
| `MMEN_BEST_test_latest.pkl` | MLP / MMEN |

**Submission ZIPs** (saved to `EK100_MIR/submission_zips/`):

| File | Contents |
|---|---|
| `ensemble_submission.zip` | Ensemble of three models (no re-ranking). |
| `reranked_submission.zip` | Ensemble + graph-diffusion re-ranking (k=20, α=0.3). |
| `JPoSE_submission.zip` | JPoSE alone. |
| `MI-MM_submission.zip` | MI-MM alone. |
| `MLP_submission.zip` | MLP / MMEN alone. |

Each ZIP contains a single `test.pkl` serialized with pickle protocol 2 and a numpy namespace compatibility patch applied (required by the Codabench grader, which runs an older numpy).

### Results

| Variant | nDCG AVG (Codabench) |
|---|---|
| Ensemble only (iteration 2) | **55.31** |
| Ensemble + re-ranking, k=20, α=0.3 (iteration 3) | **55.82** |

Re-ranking adds ~0.5 nDCG on top of the ensemble baseline. The re-ranked submission is always recommended.

---

## **Approach 3 — AVION ViT-L Fine-tuned with SMS Loss**

*Covers iterations 4, 5, and 6.*

### Strategy

Fine-tunes the **AVION** video-language model (CLIP ViT-L backbone pre-trained on Ego4D via LaViLa) with **SMS Loss** (Symmetric Multi-Similarity Loss) on the EPIC-KITCHENS-100 retrieval training set. SMS Loss uses the ground-truth soft relevancy matrix to pull together positive pairs and push apart negatives in proportion to their relevance scores, providing a more nuanced supervision signal than binary contrastive losses.

**Step 1 — Base fine-tuning (10 epochs).** The LaViLa ViT-L checkpoint is fine-tuned from scratch for 10 epochs using `ammplus_finetune.py` from the (patched) SMS-Loss repository:

```
torchrun --nproc_per_node=1 scripts/ammplus_finetune.py
  --model           CLIP_VITL14
  --batch-size      48
  --epochs          10
  --lr              2e-5
  --loss-margin     0.6   # SMS margin (θ in the paper)
  --loss-thres      0.1   # relevancy threshold; pairs below this are ignored
  --use-fast-conv1
  --grad-checkpointing    # trades compute for VRAM; required for ViT-L on a single GPU
  [--use-flash-attn]      # optional; halves attention memory, enabled when available
```

A numbered checkpoint `checkpoint_{epoch:04d}.pt` is saved after every epoch. At the end of the base run, the epoch-10 checkpoint is also duplicated as `checkpoint_round_1.pt` for use as the starting point of Step 2.

**Step 2 — Resume fine-tuning (+6 epochs, optional).** Training is extended for **6 additional epochs** (from epoch 10 to epoch 16) by passing both `--pretrain-model` and `--resume` pointing to `checkpoint_round_1.pt`, setting `--start-epoch 10`, and `--epochs 16`. Iteration 6 in the table above uses this two-phase schedule: *10 base + 6 resumed = 16 total*. The phrasing "10 + 6 epochs" reflects how the run is actually executed in the notebook, not 16 epochs trained in one shot.

**Step 3 — Inference.** `test_mir.py` encodes all test video segments and text queries, computes the full similarity matrix, and writes `submission.pkl`. Two inference settings were compared:

| Setting | Flags | Effect |
|---|---|---|
| Standard | *(none)* | Single forward pass, default 8-frame clips. |
| TTA | `--flip --clip-length 32` | Averages forward + horizontally flipped features; samples 32 frames per clip for richer temporal context. |

TTA adds ~0.7 nDCG at no training cost.

> **Same-GPU constraint:** `flash-attn` is compiled against a specific PyTorch + CUDA + GPU-architecture combination. Both fine-tuning and inference must run on the **same Colab runtime type** (same GPU class) — if you train on an A100 and then switch to a T4 for inference, you will need to either rebuild flash-attn from source on the new device or disable it.

### Notebooks (run in order)

| Notebook | Sections | Purpose |
|---|---|---|
| `src/sms_avion/setup.ipynb` | 1 | Mount Drive, configure project paths, validate or extract the EK-100 video folder, validate or extract the SMS Loss Custom repo, download the AVION pretrain checkpoint, download annotation CSVs and the train-split relevancy pickle. |
| `src/sms_avion/train_and_test.ipynb` | 1–7 | GPU verification (§1) → pinned PyTorch + CUDA + flash-attn install (§2) → Drive mount (§3) → base fine-tuning of 10 epochs (§4) → resume fine-tuning for +6 epochs (§5) → inference (§6) → submission ZIP (§7). |

`setup.ipynb` only needs to be run **once per Drive session**. `train_and_test.ipynb` is structured so that sections §1–§3 always run at the start of a Colab session, after which you only run the sections that match your workflow (training, continue training, inference only, or repackaging). See the README's *"Choose the workflow that matches your goal"* table for the exact section sequence for each scenario.

### Data & Models

| Item | Size | Notes |
|---|---|---|
| EK-100 videos (`EK100_320p_15sec_30fps_libx264`) | ~50 GB | 320p, 15-second chunks at 30 fps encoded with libx264. The user should **add a Drive shortcut** from the [public link](https://drive.google.com/file/d/13J2uC2g2H_DEHrBvgr5Aiu0BgqlCvWqG/view) — `setup.ipynb` detects either the folder form or a `EK100_320p_15sec_30fps_libx264.zip` and extracts the ZIP on first run if necessary. |
| SMS Loss Custom repo (`SMS_Loss_Custom`) | ~3 MB | Pre-patched fork of `xqwang14/SMS-Loss`; shipped at `data/SMS_Loss_Custom.zip` in this repository. Must be uploaded to Drive (as a folder or as `MyDrive/SMS_Loss_Custom.zip`) before running `setup.ipynb`. |
| AVION pretrain checkpoint (`avion_pretrain_lavila_vitl_best.pt`) | ~4.77 GB | Downloaded automatically by `setup.ipynb` from the UT Austin Box link provided by the AVION authors. |
| `EPIC_100_retrieval_train.csv` | small | Downloaded automatically by `setup.ipynb` from the EPIC-KITCHENS GitHub annotations repo. |
| `EPIC_100_retrieval_test.csv` | small | Downloaded automatically by `setup.ipynb` from the EPIC-KITCHENS GitHub annotations repo. |
| `caption_relevancy_EPIC_100_retrieval_train.pkl` | small | Soft relevancy labels for the training split; downloaded automatically by `setup.ipynb` from the LaViLa Facebook AI public files. |
| `caption_relevancy_EPIC_100_retrieval_test.pkl` | 89 MB | Soft relevancy labels for the test split; **shipped at `data/` in this repository** and must be uploaded manually to `MyDrive/EK100_annotations/` (not publicly hosted). |

> **Drive footprint:** ~50 GB of videos + ~4.77 GB of pretrain + ~16 × 5 GB of per-epoch checkpoints ≈ **~100 GB** total Drive space if you keep every checkpoint of a full reproduction. You can free space by deleting intermediate `checkpoint_0001.pt … checkpoint_0009.pt` once `checkpoint_round_1.pt` is saved.

### SMS Loss Custom Repo — Patches Applied

The SMS Loss Custom repo is a version of `xqwang14/SMS-Loss` with all required patches already applied. The patches fix the following bugs that prevented training and inference from running on the EK-100 data:

| File | Patch |
|---|---|
| `scripts/ammplus_finetune.py` | Unpack `(images, texts)` tuple correctly in the gradient-accumulation branch; replace undefined `args.accum_freq` reference with `args.update_freq`; add safe fallbacks for missing fields in `checkpoint['args']`; save a numbered `checkpoint_{epoch:04d}.pt` file at the end of every epoch; disable live validation (not needed during training, significantly slows each epoch). |
| `avion/data/clip_dataset.py` | Retry `__getitem__` up to 20 times with a random index when `get_raw_item()` returns `None` (handles corrupt or missing video chunks without crashing); guard the `relevancy_mat` load when the test relevancy path is absent. |
| `scripts/test_mir.py` | Guard against missing `checkpoint['args']` dict; strip `module.` prefix from state-dict keys when loading from a `DataParallel`-wrapped checkpoint; make `--relevancy-path` optional so a submission is always written even without the test relevancy file; skip unused train-dataset construction to avoid errors when only test data is needed. |

### Dependency Stack

Installed in `train_and_test.ipynb` (Section 2). The versions below are pinned because `flash-attn` must be compiled against a specific PyTorch + CUDA combination:

| Package | Version |
|---|---|
| `torch` | 2.4.1+cu121 |
| `torchvision` | 0.19.1+cu121 |
| `torchaudio` | 2.4.1+cu121 |
| `flash-attn` | Latest compatible with torch 2.4.1 (compiled from source, ~5 min) |
| `einops`, `kornia`, `timm`, `transformers`, `decord`, `ninja` | Latest |
| `openai/CLIP`, `open_clip_torch`, `reranking` | Latest |

`flash-attn` is optional — training and inference still run without it but are slower and consume more VRAM. **A100 (40 or 80 GB) is recommended for ViT-L fine-tuning**; T4 (16 GB) may require reducing `--batch-size` to 24–32. Because flash-attn is compiled against the device on which §2 ran, inference (§6) should be launched on the same Colab GPU class as fine-tuning.

### Files Generated

All files are saved to `experiments/sms_vitl/` on Google Drive (`EXP_DIR`):

| File | Description |
|---|---|
| `checkpoint_{epoch:04d}.pt` | Full checkpoint per epoch (model weights + optimizer state + scaler state). |
| `checkpoint_round_1.pt` | Checkpoint at the end of the base fine-tuning run (epoch 10); used as the starting point for §5 (resume fine-tuning). |
| `submission.pkl` | Raw similarity matrix (9668 × 3842) with `vis_ids` and `txt_ids`; produced by §6 (inference). |
| `submission.zip` | Codabench-compatible ZIP containing `test.pkl` (protocol-2 pickle with numpy compat patch applied); produced by §7. |

### Results

| Iteration | Training | Inference flags | nDCG AVG (Codabench) |
|---|---|---|---|
| 4 | 10 epochs from pretrain | Standard (no TTA) | **68.80** |
| 5 | 10 epochs from pretrain | `--flip --clip-length 32` | **69.48** |
| 6 | 10 + 6 epochs from pretrain (resumed from `checkpoint_round_1.pt`) | `--flip --clip-length 32` | **69.68** |

Test-time augmentation (`--flip --clip-length 32`) adds **+0.68 nDCG** at iteration 5 vs. 4 with no additional training. Continuing fine-tuning for 6 more epochs on top of the 10-epoch base adds a further **+0.20 nDCG** at iteration 6.

---

## **Reproducing Each Iteration**

The table below maps each leaderboard iteration to the exact notebook sections required to reproduce it. Section numbers refer to the headings inside each notebook.

| Iter. | nDCG AVG | Notebooks | Sections to run |
|---|---|---|---|
| 1 | 53.53 | `jpose_base_data.ipynb` → `jpose_base.ipynb` | 1–2 → 1–9 |
| 2 | 55.31 | `jpose_ensemble_data.ipynb` → `jpose_ensemble.ipynb` | 1–2 → 1–9; pick `ensemble_submission.zip` |
| 3 | 55.82 | `jpose_ensemble_data.ipynb` → `jpose_ensemble.ipynb` | 1–2 → 1–9; pick `reranked_submission.zip` |
| 4 | 68.80 | `setup.ipynb` → `train_and_test.ipynb` | full → 1, 2, 3, 4, 6, 7 (no TTA flags in §6) |
| 5 | 69.48 | `setup.ipynb` → `train_and_test.ipynb` | full → 1, 2, 3, 4, 6, 7 (with `--flip --clip-length 32` in §6) |
| 6 | 69.68 | `setup.ipynb` → `train_and_test.ipynb` | full → 1, 2, 3, 4, 5, 6, 7 (with `--flip --clip-length 32` in §6) |

For higher-level guidance on which sections to run for non-reproduction workflows (inference-only from an existing checkpoint, continue training from your own starting epoch, repackage submission only), see the README's *"Choose the workflow that matches your goal"* table.
