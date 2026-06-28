# EyeBench: Predictive Modeling from Eye Movements in Reading

[![paper](https://img.shields.io/static/v1?label=paper&message=NeurIPS%20paper&color=brightgreen)](https://openreview.net/pdf?id=LhbYJJ3MFd)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Ruff](https://github.com/EyeBench/eyebench/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/EyeBench/eyebench/actions/workflows/ruff.yml)

---

![EyeBench Overview](./docs/figures/eyebench_overview.png)
*Figure 1: Overview of EyeBench v1.0. The benchmark curates multiple datasets for predicting reader properties (👩), and reader–text interactions (👩+📝) from eye movements. ⭐ marks prediction tasks newly introduced in EyeBench. The data are preprocessed and standardized into aligned text and gaze sequences, which are then used as input to models trained to predict task-specific targets. The models are systematically evaluated under three generalization regimes — unseen readers, unseen texts, or both. The benchmark supports the evaluation and addition of new models, datasets, and tasks.*

---

## 🧠 Introduction

**EyeBench** is the first benchmark designed to evaluate machine learning models that decode cognitive and linguistic information from eye movements during reading.  
It provides a standardized, extensible framework for **predictive modeling from eye tracking data**, aiming to bridge **cognitive science and multimodal AI**.

EyeBench curates multiple publicly available datasets and tasks, covering both **reader properties** and **reader–text interactions**, and includes **baselines**, **state-of-the-art models**, and **evaluation protocols** that ensure reproducibility and comparability across studies.

Progress on EyeBench is expected to advance both **scientific understanding of human language processing** and **practical applications** such as adaptive educational systems and cognitive-aware user interfaces.

Official repository: [https://github.com/EyeBench/eyebench](https://github.com/EyeBench/eyebench)

---

## 📚 Tasks and Datasets

EyeBench v1.0 includes **seven prediction tasks** spanning **six harmonized datasets**.  
Each task is formulated as a **single-trial prediction problem** from a reader’s eye movements while reading a passage (and optionally an auxiliary text, such as a question or claim).

### Reader Properties (👤)

| Task | Dataset | Type | Target |
|------|----------|------|--------|
| **Reading Comprehension Skill** | CopCo | Regression | Continuous comprehension score (1–10) |
| **Vocabulary Knowledge** | MECO L2 | Regression | LexTALE vocabulary test score (0–100) |
| **Dyslexia Detection** | CopCo | Classification | Clinically diagnosed dyslexia (yes/no) |

### Reader–Text Interactions (👤 + 📖)

| Task | Dataset(s) | Type | Target |
|------|-------------|------|--------|
| **Reading Comprehension** | OneStop, SB-SAT, PoTeC | Classification | Correct answer to a comprehension question |
| **Subjective Text Difficulty** | SB-SAT | Regression | Perceived difficulty rating (Likert) |
| **Domain Expertise** | PoTeC | Classification | High vs low domain expertise |
| **Claim Verification** | IITB-HGC | Classification | Correct claim verification judgment |

### Datasets Overview

| Dataset | Language | Group | #Participants | #Words | #Fixations | Tasks |
|----------|-----------|--------|----------------|----------|-------------|--------|
| OneStop (Ordinary Reading) | English | L1 | 180 | 19 427 | 1.1 M | Reading Comprehension |
| SB-SAT | English | L1/L2 | 95 | 2 622 | 263 k | Reading Comprehension, Subjective Text Difficulty |
| PoTeC | German | L1 | 75 | 1 895 | 404 k | Reading Comprehension, Domain Expertise |
| MECO L2 | English | L2 | 1 098 | 1 646 | 2.4 M | Vocabulary Knowledge |
| CopCo | Danish | L1/L2/L1-Dyslexia | 57 | 32 140 | 398 k | Reading Comprehension Skill, Dyslexia Detection |
| IITB-HGC | English | L1/L2 | 5 | 53 528 | 164 k | Claim Verification |

---

## 🧩 Implemented Models and Baselines

EyeBench provides **12 implemented models** and **6 baselines**, unified under a shared training and evaluation framework.

### Neural Models

- **AhnCNN** – CNN over fixation sequences (coordinates, durations, pupil size)  
- **AhnRNN** – RNN variant of AhnCNN  
- **BEyeLSTM** – LSTM combining sequential fixations and global gaze statistics  
- **PLM-AS** – LSTM processing fixation-ordered word embeddings  
- **PLM-AS-RM** – RNN integrating fixation-ordered embeddings with reading measures  
- **RoBERTEye-W** – Transformer integrating word embeddings and word-level gaze features  
- **RoBERTEye-F** – Fixation-level variant of RoBERTEye-W  
- **MAG-Eye** – Multimodal Adaptation Gate injecting gaze into transformer layers  
- **PostFusion-Eye** – Cross-attention fusion of RoBERTa embeddings and CNN fixation features  

### Traditional ML Models

- **Logistic / Linear Regression**  
- **Support Vector Machine (SVM / SVR)**  
- **Random Forest (Classifier / Regressor)**  

### Baselines

- **Random** and **Majority Class** (classification)  
- **Mean** and **Median** (regression)  
- **Reading Speed**  
- **Text-Only RoBERTa** (no gaze input)

---

## 🧮 Evaluation Protocol

EyeBench evaluates models under **three complementary generalization regimes**:

| Regime | Description | Typical Use Case |
|---------|-------------|------------------|
| **Unseen Reader** | Texts seen, readers unseen | New readers, known materials |
| **Unseen Text** | Readers seen, texts unseen | Personalized reading of new content |
| **Unseen Reader & Text** | Both unseen | Fully general setting |

### Metrics

- **Classification:** AUROC, Balanced Accuracy  
- **Regression:** RMSE, MAE, R²  
- **Aggregate:** Average Normalized Score and Mean Rank across all task–dataset pairs.

---

## ⚙️ Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/EyeBench/eyebench.git
cd eyebench
mamba env create -f environment.yml
conda activate eyebench
```

### 2. Download and Preprocess Data

```bash
bash src/data/preprocessing/get_data.sh
```

This script downloads, harmonizes, and creates standardized folds for all datasets under `data/processed/`.

### 3. Log into Weights & Biases (WandB)

```bash
wandb login
```

---

## 🚀 Usage

### Train a Model

```bash
python src/run/single_run/train.py +trainer=TrainerDL +model=RoberteyeWord +data=OneStop_RC
```

### Run a Hyperparameter Sweep

```bash
bash run_commands/utils/sweep_wrapper.sh --data_tasks CopCo_TYP --folds 0,1,2,3 --cuda 0,1
```

### Test a Model

```bash
python src/run/single_run/test_dl.py +model=RoberteyeWord +data=OneStop_RC
```

Results are stored under:
`
results/raw/{data_model_trainer_task}/fold_index={i}/trial_level_test_results.csv
results/eyebench_benchmark_results/{metric}.csv
`

---

## 🧠 Adding a New Model

1. Create a file under `src/models/YourModel.py` inheriting from `BaseModel`.
   Implement `forward()` and `shared_step()` methods.
2. Register it in:

    - `src/configs/constants.py` → `DLModelNames` (or `MLModelNames` for ML models)
    - `src/configs/models/dl/YourModel.py` → model config class decorated with `@register_model_config` (use `src/configs/models/ml/` for ML models)

3. Define its default parameters and search space in `src/run/multi_run/search_spaces.py`.
4. Verify integration:

```bash
bash run_commands/utils/model_checker.sh
```

---

## 📊 Adding a New Dataset

1. Store raw or preprocessed data in `data/YOUR_DATASET/`.
2. Define its loading logic in `src/data/datasets/YOUR_DATASET.py` (inherits from `ETDataset`).
3. Add preprocessing logic under `src/data/preprocessing/dataset_preprocessing/YOUR_DATASET.py`.
4. Register the dataset in `src/configs/data.py` and `src/configs/constants.py`.
5. Add a corresponding task configuration class if it supports multiple tasks.

Datasets must comply with EyeBench’s selection criteria:

- Passage-level texts
- ≥ 500 Hz sampling rate
- Publicly available raw or fixation-level data
- Released texts and gaze–text alignment

---

## 📘 Documentation

To build the local documentation site:

```bash
pip install mkdocs mkdocs-material 'mkdocstrings[python]' mkdocs-gen-files mkdocs-literate-nav
mkdocs serve
```

---

## 📄 Citation

If you use EyeBench in your research, please cite:

> Omer Shubi, David R. Reich, Keren Gruteke Klein, Yuval Angel, Paul Prasse, Lena Jäger, Yevgeni Berzak.
> **EyeBench: Predictive Modeling from Eye Movements in Reading.**
> *NeurIPS 2025.*

```bibtex
@inproceedings{shubireich2025eyebench,
  title={{EyeBench}: {P}redictive Modeling from Eye Movements in Reading},
  author={Shubi, Omer and Reich, David Robert and Gruteke Klein, Keren and Angel, Yuval and Prasse, Paul and J{\"a}ger, Lena Ann and Berzak, Yevgeni},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=LhbYJJ3MFd}
}
```

---

## 🤝 Acknowledgments

EyeBench development is supported by:

- **COST Action MultiplEYE (CA21131)**
- **Swiss National Science Foundation (EyeNLG, IZCOZ0 _220330)**
- **Israel Science Foundation (grant 1499/22)**

---

## 🧩 License

All datasets included in EyeBench follow their respective original licenses.
Code released under the [MIT License](LICENSE).

---

## Cloud Run With Processed HF Data

This fork is set up so GPU instances do not need to download or preprocess raw EyeBench data. The helper below downloads only the processed EyeBench folders required by the existing training and evaluation code:

- `data/<dataset>/processed/`
- `data/<dataset>/folds/`
- `data/<dataset>/folds_metadata/`

### PoTeC_RC MAG Baseline

```bash
cd /workspace/mae_foreyebench
conda activate eyebench

# Do not use `pip install -U huggingface_hub...` here. It can upgrade
# dependencies beyond the versions expected by `datasets==3.5.0` and
# `lightning==2.5.1.post0`.
python -m pip install --force-reinstall \
  "huggingface_hub[hf_xet]==0.36.2" \
  "fsspec[http]==2024.12.0" \
  "dill==0.3.8" \
  "multiprocess==0.70.16" \
  "packaging==24.2"

python - <<'PY'
import torch
import transformers
import datasets
import huggingface_hub
import fsspec
import dill
import multiprocess
import packaging
import lightning

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("fsspec", fsspec.__version__)
print("dill", dill.__version__)
print("multiprocess", multiprocess.__version__)
print("packaging", packaging.__version__)
print("lightning", lightning.__version__)
PY

hf auth login

python run_commands/utils/download_processed_folds_from_hf.py \
  skboy/eyebench-processed-folds \
  --local-dir . \
  --datasets PoTeC

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
bash run_commands/run_potec_rc_baseline_from_hf.sh \
  2>&1 | tee logs/run_potec_rc_mag_baseline_$(date +%Y%m%d_%H%M).log
```

The default script downloads PoTeC from `skboy/eyebench-processed-folds`, trains the EyeBench `MAG` baseline on folds `0 1 2 3`, runs `test_dl.py`, and writes a result archive at the repository root.

Expected core versions after the dependency repair step:

```text
torch 2.5.1
cuda True
transformers 4.47.1
datasets 3.5.0
huggingface_hub 0.36.2
fsspec 2024.12.0
dill 0.3.8
multiprocess 0.70.16
packaging 24.2
lightning 2.5.1.post0
```

To run several baselines:

```bash
BASELINE_MODELS="MAG RoberteyeWord Roberta" \
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
bash run_commands/run_potec_rc_baseline_from_hf.sh
```

To download all processed benchmark datasets without training:

```bash
python run_commands/utils/download_processed_folds_from_hf.py \
  skboy/eyebench-processed-folds \
  --local-dir . \
  --datasets all
```

### C-MAE-MAG-Eye Experimental Model

This fork adds a modular C-MAE implementation under `model/` and registers a Hydra model config named `CMAEMAGEye`.
The model keeps the existing MAG baseline path, adds a text-conditioned masked gaze autoencoder, and feeds the C-MAE trial representation into the final logits through a small gated residual head.

Smoke run for one fold:

```bash
cd /workspace/mae_foreyebench
conda activate eyebench
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python src/run/single_run/train.py \
  +trainer=TrainerDL \
  +model=CMAEMAGEye \
  +data=PoTeC_RC \
  data.fold_index=0 \
  trainer.devices=1 \
  trainer.precision=THIRTY_TWO_TRUE \
  trainer.num_workers=4 \
  trainer.wandb_entity=offline \
  trainer.wandb_project=CMAEMAG_PoTeC_RC_smoke \
  trainer.wandb_job_type=CMAEMAGEye_PoTeC_RC_fold0 \
  model.batch_size=2 \
  model.accumulate_grad_batches=8 \
  model.max_epochs=1 \
  model.early_stopping_patience=1 \
  model.mag_injection_index=23 \
  model.cmae_reconstruction_loss_weight=0.1 \
  hydra.run.dir='outputs/cmaemageye_smoke/fold_index=0'
```

The existing baseline runner can also launch it:

```bash
BASELINE_MODELS="CMAEMAGEye" \
DATA_TASK=PoTeC_RC \
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
bash run_commands/run_potec_rc_baseline_from_hf.sh
```
