# Multi-Instance Retrieval — EPIC-KITCHENS-100

Solution for the [Codabench EK-100 Multi-Instance Retrieval](https://www.codabench.org/competitions/12008) competition.

Given a natural-language query (e.g. *"cut tomato"*), the task is to rank **9,668 video segments** by relevance, and vice versa. Performance is measured by **nDCG AVG**, the average of the video-to-text and text-to-video normalized Discounted Cumulative Gain.

This repository contains three progressively stronger approaches, implemented as Google Colab notebooks. The best approach reaches **69.68 nDCG AVG** on the public Codabench leaderboard.

### Interactive demo (Hugging Face)

- **Gradio Space:** [jsurrea/ek100-mir-demo](https://huggingface.co/spaces/jsurrea/ek100-mir-demo) — preset benchmark queries, random video→text clips, and optional free-text search (CPU).
- **Model:** [jsurrea/avion-vitl-ek100-sms](https://huggingface.co/jsurrea/avion-vitl-ek100-sms)
- **Assets dataset:** [jsurrea/ek100-mir-demo-assets](https://huggingface.co/datasets/jsurrea/ek100-mir-demo-assets)

---

## **Results**

| # | Approach | nDCG AVG (Codabench) |
|---|---|---|
| 1 | JPoSE Base | 53.53 |
| 2 | JPoSE Ensemble (JPoSE + MI-MM + MLP) | 55.31 |
| 3 | JPoSE Ensemble + graph-diffusion re-ranking | 55.82 |
| 4 | AVION ViT-L + SMS Loss (10 epochs, standard inference) | 68.80 |
| 5 | AVION ViT-L + SMS Loss (10 epochs, flip + 32-frame TTA) | 69.48 |
| **6** | **AVION ViT-L + SMS Loss (10 + 6 epochs, flip + 32-frame TTA)** | **69.68** |

A full breakdown of each iteration — strategy, hyperparameters, data sources, and patches applied — is in [`APPROACHES.md`](APPROACHES.md).

---

## **Repository Layout**

```
.
├── APPROACHES.md                   Detailed write-up of every iteration
├── README.md                       This file
├── LICENSE                         MIT
├── data/                           Files that must be uploaded manually to Drive
│   ├── SMS_Loss_Custom.zip         Pre-patched fork of SMS-Loss (Approach 3)
│   └── caption_relevancy_EPIC_100_retrieval_test.pkl   Soft test relevancies (89 MB)
├── src/
│   ├── jpose_base/                 Approach 1 — JPoSE Base
│   │   ├── jpose_base_data.ipynb       Step 1: download data
│   │   └── jpose_base.ipynb            Step 2: inference + submission
│   ├── jpose_ensemble/             Approach 2 — JPoSE + MI-MM + MLP ensemble
│   │   ├── jpose_ensemble_data.ipynb   Step 1: download data
│   │   └── jpose_ensemble.ipynb        Step 2: inference, ensemble, re-rank, submission
│   └── sms_avion/                  Approach 3 — AVION ViT-L fine-tuned with SMS Loss
│       ├── setup.ipynb                 Step 1 (run once): validate data, download pretrain
│       └── train_and_test.ipynb        Step 2: install deps, fine-tune, infer, submission
├── space/                          Hugging Face Gradio demo (runtime assets on HF dataset)
│   ├── app.py                      UI entrypoint
│   ├── config.py, rank.py, …       Ranking, loaders, SMS/AVION hooks
│   └── requirements.txt            Space dependencies
└── archive/
    └── sms_avion_base.ipynb        Legacy monolithic version of Approach 3 (reference only)
```

---

## **Prerequisites**

The project is designed end-to-end for **Google Colab + Google Drive**. No local Python environment is required — all dependencies are installed from the notebooks.

You will need:

- A Google account with **Google Drive**. The pre-processed EK-100 videos needed for Approach 3 weigh **~50 GB** and live on a public Google Drive — you will **add a shortcut** to your own Drive (see [Approach 3 → Required data](#approach-3---required-data-on-drive)), so you do *not* need to upload 50 GB yourself. However, to reproduce the best result, you need **~100 GB** storage in Google Drive. 
- A **Colab session with a GPU**:
  - Approaches 1 & 2: any GPU runtime (T4 is fine, inference only).
  - Approach 3: **A100 (40 GB) recommended** for fine-tuning the ViT-L backbone. Due to **flash attention** dependencies, the inference should be run in the same device or GPU.

---

## **How to Run — Detailed Per-Notebook Walkthrough**

Each notebook is internally split into numbered sections. The tables below say exactly which sections to run depending on the workflow you want.

### Approach 1 — JPoSE Base (any GPU)

#### `src/jpose_base/jpose_base_data.ipynb`

Run **all sections** once per Drive.

| Section | What it does |
|---|---|
| 1. Google Drive Mount | Mounts Drive, declares paths under `MyDrive/EK100_MIR/`. |
| 2. Data Download | Downloads `JPoSE_data.zip` (~1.64 GB) and the EPIC-100 retrieval annotation PKLs. |

#### `src/jpose_base/jpose_base.ipynb`

Run **all sections** every time you want a new submission.

| Section | What it does |
|---|---|
| 1. GPU Environment Verification | `nvidia-smi` check. |
| 2. Google Drive Mount | Re-mounts Drive (notebook can be re-entered cold). |
| 3. Dependencies Installation | Installs pinned versions of PyTorch and JPoSE deps. |
| 4. Repository Setup | Clones the [JPoSE](https://github.com/mwray/Joint-Part-of-Speech-Embeddings) repo. |
| 5. Paths Configuration | Resolves all input/output paths on Drive. |
| 6. Data Extraction | Extracts `JPoSE_data.zip` into the JPoSE repo tree. |
| 7. Compatibility Patches | Regex-patches every `torch.load(…)` call to include `weights_only=False` (required for PyTorch ≥ 2.0). |
| 8. Inference | Runs JPoSE inference and writes the raw similarity matrix to `EK100_MIR/submissions/JPoSE_BEST_test_latest.pkl`. |
| 9. Build Submission Zip | Packages `JPoSE_BEST_submission.zip` into `EK100_MIR/submission_zips/`. |

**Upload** `JPoSE_BEST_submission.zip` to Codabench.

---

### Approach 2 — JPoSE Ensemble + Re-ranking (any GPU)

#### `src/jpose_ensemble/jpose_ensemble_data.ipynb`

Run **all sections** once per Drive.

| Section | What it does |
|---|---|
| 1. Google Drive Mount | Mounts Drive, declares paths under `MyDrive/EK100_MIR/`. |
| 2. Data Download | Downloads `JPoSE_data.zip` (~1.64 GB), `MI-MM_data.zip` (~0.68 GB), and the EPIC-100 annotation PKLs. |

#### `src/jpose_ensemble/jpose_ensemble.ipynb`

Run **all sections** end-to-end.

| Section | What it does |
|---|---|
| 1. GPU Environment Verification | `nvidia-smi` check. |
| 2. Google Drive Mount | Re-mounts Drive. |
| 3. Dependencies Installation | Installs pinned versions of PyTorch and JPoSE / MI-MM deps. |
| 4. Repository Setup | Clones both the JPoSE and MI-MM repos. |
| 5. Paths Configuration | Resolves all input/output paths on Drive. |
| 6. Data Extraction | Extracts `JPoSE_data.zip` and `MI-MM_data.zip`. |
| 7. Compatibility Patches | Applies `weights_only=False` patches to JPoSE and MI-MM; replaces `pickle5` with `pickle` in MI-MM. |
| 8. Inference | Runs JPoSE, MI-MM, and MLP/MMEN inference. Computes min-max-normalized equal-weight ensemble. Applies graph-diffusion re-ranking (k = 20, α = 0.3). |
| 9. Build Submission Zip | Writes **five** ZIPs to `EK100_MIR/submission_zips/`: `reranked_submission.zip` (best — 55.82), `ensemble_submission.zip`, `JPoSE_submission.zip`, `MI-MM_submission.zip`, `MLP_submission.zip`. |

**Upload** `reranked_submission.zip` to Codabench.

---

### Approach 3 — AVION ViT-L + SMS Loss

This is the best-performing pipeline. It has two notebooks: a one-time `setup.ipynb` and a multi-mode `train_and_test.ipynb`.

#### Approach 3 — Required data on Drive

**Before running anything**, prepare the following on Google Drive:

| Item | What to do |
|---|---|
| **EK-100 videos** (`EK100_320p_15sec_30fps_libx264`, ~50 GB) | Open this public Drive folder/ZIP: [Drive shortcut link](https://drive.google.com/file/d/13J2uC2g2H_DEHrBvgr5Aiu0BgqlCvWqG/view) and click **Add shortcut to Drive → MyDrive**. *Do not download and re-upload* — a shortcut is instant. `setup.ipynb` will auto-detect both the folder (`MyDrive/EK100_320p_15sec_30fps_libx264/`) and the ZIP (`MyDrive/EK100_320p_15sec_30fps_libx264.zip`) form, and extracts the ZIP on first run if needed. |
| **SMS Loss Custom** repo (shipped at `data/SMS_Loss_Custom.zip` in this repo, ~3 MB) | Upload as `MyDrive/SMS_Loss_Custom.zip` (or as the extracted folder `MyDrive/SMS_Loss_Custom/`). `setup.ipynb` will unzip it on first run if needed. |
| **Test relevancy pickle** (shipped at `data/caption_relevancy_EPIC_100_retrieval_test.pkl` in this repo, 89 MB) | Upload to `MyDrive/EK100_annotations/caption_relevancy_EPIC_100_retrieval_test.pkl`. This file is not publicly hosted, so it must be uploaded manually. |

All other large files (AVION pretrain checkpoint, annotation CSVs, train relevancy pickle) are downloaded automatically by `setup.ipynb`.

#### `src/sms_avion/setup.ipynb` — run **once per Drive** 

This notebook has a single section (*1. Data Download and Paths Configuration*); run every cell top-to-bottom. It will:

1. Mount Drive and declare project paths.
2. Validate the EK-100 video data — extract the ZIP if the folder is not yet present.
3. Validate the SMS Loss Custom repo — extract the ZIP if the folder is not yet present.
4. Download the AVION LaViLa ViT-L pretrain checkpoint `avion_pretrain_lavila_vitl_best.pt` (~4.77 GB) into `MyDrive/checkpoints/`.
5. Download EPIC-100 retrieval annotation CSVs and the train-split relevancy pickle into `MyDrive/EK100_annotations/`.
6. Verify the (manually uploaded) test-split relevancy pickle is present.

Re-running it is safe — every download/extract step is idempotent.

#### `src/sms_avion/train_and_test.ipynb` — depends on what you want to do

This notebook is organized into seven sections. **Sections 1–3 must run at the start of every Colab session** (they install dependencies and mount Drive). After that, you only run the sections that match your workflow:

| Section | What it does |
|---|---|
| 1. GPU Environment Verification | `nvidia-smi` and CUDA sanity check. |
| 2. Dependencies Installation | Pinned PyTorch 2.4.1+cu121, `flash-attn`, `decord`, `kornia`, `timm`, `transformers`, etc. |
| 3. Google Drive Mount | Re-mounts Drive and re-declares all project paths. |
| **4. Base Fine-tuning** | Fine-tunes the AVION ViT-L pretrain for **10 epochs** with SMS Loss (`--batch-size 48 --lr 2e-5 --loss-margin 0.6 --loss-thres 0.1`). Writes `checkpoint_0001.pt … checkpoint_0010.pt` plus `checkpoint_round_1.pt` to `MyDrive/experiments/sms_vitl/`. |
| **5. Resume Fine-tuning from Checkpoint** | Continues training from `checkpoint_round_1.pt` (or any other checkpoint you point at) up to **epoch 16**. Adds +0.20 nDCG. |
| **6. Inference over Test** | Runs `test_mir.py` with `--flip --clip-length 32` (TTA) by default. Writes `submission.pkl` to `MyDrive/experiments/sms_vitl/`. |
| **7. Build Submission Zip** | Packages `submission.pkl` as `submission.zip` with the numpy-namespace compatibility patch applied. |

**Choose the workflow that matches your goal:**

| Goal | Sections to run | Notes |
|---|---|---|
| **Reproduce the 69.68 leaderboard result (full pipeline)** | 1 → 2 → 3 → 4 → 5 → 6 → 7 | ~24 h total on A100. |
| Reproduce only the 68.80 / 69.48 variants (10-epoch model) | 1 → 2 → 3 → 4 → 6 → 7 | Skip Section 5. In Section 6, set `--flip --clip-length 32` for 69.48 or no flags for 68.80. |
| **Train from scratch only** (no inference) | 1 → 2 → 3 → 4 (and optionally 5) | Final checkpoints land in `MyDrive/experiments/sms_vitl/`. |
| **Continue training only** from a checkpoint you already have | 1 → 2 → 3 → 5 | Make sure your starting checkpoint sits at `MyDrive/experiments/sms_vitl/checkpoint_{epoch:04d}.pt` and adjust `--start-epoch` / `--epochs` in Section 5 to match your starting and target epochs. |
| **Inference only** from an existing checkpoint | 1 → 2 → 3 → 6 → 7 | Place your `.pt` at `MyDrive/experiments/sms_vitl/checkpoint_{epoch:04d}.pt` and point Section 6 at it. Use `--flip --clip-length 32` for TTA. |
| **Repackage submission only** (you already have `submission.pkl`) | 1 → 3 → 7 | Dependency install (Section 2) is not needed for packaging. |

**Upload** `MyDrive/experiments/sms_vitl/submission.zip` to Codabench.

---

## **Data Used per Approach**

A consolidated view of every external file each approach consumes, where it ends up on Drive, and how to obtain it.

### Approach 1 — JPoSE Base

| File | Size | Drive destination | How to obtain |
|---|---|---|---|
| `JPoSE_data.zip` (pre-extracted features + JPoSE_BEST + MMEN_BEST checkpoints) | 1.64 GB | `MyDrive/EK100_MIR/data/JPoSE_data.zip` | Auto-downloaded from Dropbox by `jpose_base_data.ipynb` |
| `EPIC_100_retrieval_train.pkl` | ~6 MB | `MyDrive/EK100_MIR/data/annotations/` | Auto-downloaded from [epic-kitchens-100-annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) by `jpose_base_data.ipynb` |
| `EPIC_100_retrieval_test.pkl` | ~6 MB | `MyDrive/EK100_MIR/data/annotations/` | Auto-downloaded from [epic-kitchens-100-annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) by `jpose_base_data.ipynb` |

### Approach 2 — JPoSE Ensemble + Re-ranking

Everything from Approach 1, **plus**:

| File | Size | Drive destination | How to obtain |
|---|---|---|---|
| `MI-MM_data.zip` (MI-MM checkpoint epoch 202 + S3D HowTo100M features) | 0.68 GB | `MyDrive/EK100_MIR/data/MI-MM_data.zip` | Auto-downloaded from Dropbox by `jpose_ensemble_data.ipynb` |

### Approach 3 — AVION ViT-L + SMS Loss

| File | Size | Drive destination | How to obtain |
|---|---|---|---|
| EK-100 pre-processed videos (`EK100_320p_15sec_30fps_libx264`, 320p / 15 s / 30 fps / libx264) | ~50 GB | `MyDrive/EK100_320p_15sec_30fps_libx264/` (or `.zip`) | **Add a shortcut** from this public Drive: [link](https://drive.google.com/file/d/13J2uC2g2H_DEHrBvgr5Aiu0BgqlCvWqG/view) → *Add shortcut to MyDrive*. `setup.ipynb` auto-extracts the ZIP if no folder is present. |
| `SMS_Loss_Custom` (patched fork of [xqwang14/SMS-Loss](https://github.com/xqwang14/SMS-Loss)) | 3 MB | `MyDrive/SMS_Loss_Custom/` (or `.zip`) | **Manual** — shipped at `data/SMS_Loss_Custom.zip` in this repo. Upload to `MyDrive/`. |
| `caption_relevancy_EPIC_100_retrieval_test.pkl` | 89 MB | `MyDrive/EK100_annotations/caption_relevancy_EPIC_100_retrieval_test.pkl` | **Manual** — shipped at `data/caption_relevancy_EPIC_100_retrieval_test.pkl` in this repo. Not publicly hosted. |
| `avion_pretrain_lavila_vitl_best.pt` (CLIP ViT-L pre-trained on Ego4D via LaViLa) | 4.77 GB | `MyDrive/checkpoints/avion_pretrain_lavila_vitl_best.pt` | Auto-downloaded by `setup.ipynb` from the AVION authors' UT-Austin Box. |
| `EPIC_100_retrieval_train.csv`, `EPIC_100_retrieval_test.csv` | small | `MyDrive/EK100_annotations/` | Auto-downloaded by `setup.ipynb` from [epic-kitchens-100-annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations). |
| `caption_relevancy_EPIC_100_retrieval_train.pkl` | small | `MyDrive/EK100_annotations/` | Auto-downloaded by `setup.ipynb` from the [LaViLa public files](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/). |
| Sentence-level CSVs (`EPIC_100_retrieval_{train,test}_sentence.csv`) | small | `MyDrive/EK100_annotations/` | Auto-downloaded by `setup.ipynb` from [epic-kitchens-100-annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations). |

---

## **Expected Google Drive Layout**

After running everything, your Drive will look like this:

```
MyDrive/
├── EK100_MIR/                                Approaches 1 & 2 workspace
│   ├── data/
│   │   ├── JPoSE_data.zip
│   │   ├── MI-MM_data.zip                    (Approach 2 only)
│   │   └── annotations/
│   │       ├── EPIC_100_retrieval_train.pkl
│   │       └── EPIC_100_retrieval_test.pkl
│   ├── submissions/                          Raw .pkl similarity matrices
│   └── submission_zips/                      Final Codabench ZIPs
│
├── EK100_320p_15sec_30fps_libx264/           EK-100 video clips, ~50 GB (Drive shortcut)
├── EK100_annotations/                        Approach 3 annotations + relevancies
├── SMS_Loss_Custom/                          Patched SMS-Loss repo (Approach 3)
├── checkpoints/
│   └── avion_pretrain_lavila_vitl_best.pt    ~4.77 GB
└── experiments/sms_vitl/                     Approach 3 outputs
    ├── checkpoint_0001.pt … checkpoint_0016.pt
    ├── checkpoint_round_1.pt
    ├── submission.pkl
    └── submission.zip
```

---

## **Submission Format**

All three approaches produce the same Codabench-compatible output:

`submission.zip` contains a single `test.pkl` (pickle protocol 2) with the following dictionary:

```python
{
  "version":   "0.1",
  "challenge": "multi_instance_retrieval",
  "sls_pt":    2,                 # supervision level — pre-training
  "sls_tl":    3,                 # supervision level — training labels
  "sls_td":    3,                 # supervision level — training data
  "sim_mat":   ndarray(9668, 3842, dtype=float32),   # similarity scores
  "vis_ids":   [str] * 9668,      # video segment IDs (row order of sim_mat)
  "txt_ids":   [str] * 3842,      # narration IDs    (column order of sim_mat)
}
```

A byte-level patch (`b"numpy._core.multiarray" → b"numpy.core.multiarray"`) is applied to the serialized pickle so that the Codabench grader (which runs on an older numpy) can unpickle modern numpy ≥ 2.0 arrays. This patch is built into every submission packager in this repo — you do not need to do anything manually.

---

## **Troubleshooting**

- **`torch.load` raises `WeightsUnpickler` errors.** The JPoSE and MI-MM source trees were written against PyTorch < 2.0. The notebooks apply an in-place regex patch that adds `weights_only=False` to every `torch.load(…)` call before running inference. If you re-extract the upstream ZIPs manually, re-run the patch cell (Section 7 of the JPoSE inference notebooks).
- **Codabench rejects the submission with a numpy unpickling error.** The submission ZIPs must contain a pickle that uses the legacy `numpy.core.multiarray` namespace, not `numpy._core.multiarray`. The packaging cells in every notebook apply this byte-level patch automatically — re-run Section 9 (JPoSE) or Section 7 (SMS-AVION).
- **`setup.ipynb` prints `Neither folder nor zip found`.** You forgot to add the Drive shortcut for the EK-100 videos. Open the [shortcut link](https://drive.google.com/file/d/13J2uC2g2H_DEHrBvgr5Aiu0BgqlCvWqG/view), choose *Add shortcut to MyDrive*, then re-run the cell.
- **CUDA OOM during fine-tuning (Approach 3).** Lower `--batch-size` (try 32 or 24) and keep `--grad-checkpointing` enabled. `--use-flash-attn`, when available, roughly halves attention memory.
- **`flash-attn` fails to build.** It is optional. Skip the flash-attn install cell; the training and inference scripts auto-detect availability and fall back to the standard attention kernel.
- **Section 5 (resume training) starts from epoch 0.** The notebook expects both `--pretrain-model` and `--resume` to point to the same checkpoint and `--start-epoch` to match the saved epoch. Edit those flags before running Section 5 if you are resuming from a checkpoint other than `checkpoint_round_1.pt`.

---

## **References**

- **Competition:** [Codabench EK-100 Multi-Instance Retrieval (12008)](https://www.codabench.org/competitions/12008)
- **JPoSE — Joint Part-of-Speech Embeddings:** [mwray/Joint-Part-of-Speech-Embeddings](https://github.com/mwray/Joint-Part-of-Speech-Embeddings)
- **MI-MM — Multi-Instance Multi-Modal:** [adrianofragomeni/MI-MM](https://github.com/adrianofragomeni/MI-MM)
- **AVION:** [zhaoyue-zephyrus/AVION](https://github.com/zhaoyue-zephyrus/AVION)
- **LaViLa pretrain:** [facebookresearch/LaViLa](https://github.com/facebookresearch/LaViLa)
- **SMS Loss:** [xqwang14/SMS-Loss](https://github.com/xqwang14/SMS-Loss) (a patched fork is shipped at `data/SMS_Loss_Custom.zip`)
- **EPIC-KITCHENS-100 dataset & annotations:** [epic-kitchens.github.io](https://epic-kitchens.github.io/2024) — [annotations repo](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

---

## **License**

Released under the [MIT License](LICENSE). The repository ships only original code, configuration, and a patched fork of SMS-Loss; the EK-100 dataset, the AVION pretrain checkpoint, the LaViLa relevancy pickles, and the JPoSE / MI-MM / MMEN model weights are property of their respective authors and remain under their original licenses.
