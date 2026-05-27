# ML Ops Assignment 1 — g25ai1025

**Repository:** https://github.com/kst577/G25AI1025-MlLOPS  

End-to-end workflow to predict Boston house prices with **DecisionTreeRegressor** and **KernelRidge** (`scikit-learn`), reusable helpers in **`misc.py`**, and **GitHub Actions** CI on the `kernelridge` branch.

**Student / roll:** g25ai1025  
**Reproducibility seed:** `1025` (derived from roll number)

> **Submission (course):** report as PDF only — `g25ai1025 A1.pdf`; keep this repo **public** through the deadline. Do **not** delete branches (`main`, `dtree`, `kernelridge`). Use CLI for Git / push (avoid web-upload penalty).

---

## Branch structure

| Branch | Contents |
|--------|----------|
| `main` | This **README**, merged code, and runnable instructions |
| `dtree` | `requirements.txt`, `misc.py`, `train.py` (Decision Tree; test-set MSE) |
| `kernelridge` | `train2.py` (Kernel Ridge; test-set MSE) + GitHub Actions workflow |

Workflow: introduce work on **`dtree`** → merge into **`main`** → continue on **`kernelridge`**. CI is configured so that a **push to `kernelridge`** checks out the code, installs dependencies, and runs **both** `train.py` and `train2.py` to show performance.

---

## Prerequisites

- [Conda](https://docs.conda.io/) (recommended in class)  
- Git + GitHub (CLI workflow as per assignment)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kst577/G25AI1025-MlLOPS.git
cd G25AI1025-MlLOPS
```

### 2. Create and activate a conda environment

```bash
conda create -n mlops-a1 python=3.10 -y
conda activate mlops-a1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` is introduced on the **`dtree`** branch and merged into **`main`** with the rest of the code.

---

## How to run

From the project root (conda env activated):

```bash
python train.py    # DecisionTreeRegressor — displays average MSE on the test set
python train2.py   # KernelRidge — displays average MSE on the test set
```

Training scripts should call only **generic** routines in **`misc.py`** for loading, preprocessing, fitting, and evaluation so other regressors can plug in the same way.

---

## Project layout

```
.
├── misc.py                 # Shared: load_data, preprocess, train, evaluate, orchestration
├── train.py               # Decision tree experiment
├── train2.py              # Kernel ridge experiment
├── requirements.txt
├── README.md
└── .github/workflows/     # CI (push trigger on kernelridge branch)
```

---

## Dataset

Boston Housing is no longer shipped in newer `sklearn`; this project builds the dataframe as in the [sklearn load_boston](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html) docs: CSV from `http://lib.stat.cmu.edu/datasets/boston`, reshaped features + target **`MEDV`**, implemented in **`misc.load_data()`**.

Features: `CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`.

---

## Report checklist (documentation)

Include in **`g25ai1025 A1.pdf`**:

1. GitHub repo link **at the top** (same as above).  
2. **Performance comparison** (test MSE — Decision Tree vs Kernel Ridge).  
3. Screenshots of **`main`**, **`dtree`**, and **`kernelridge`** on GitHub after pushing.  
4. Screenshot of **GitHub Actions** logs (workflow run on **`kernelridge`**).

---

If you edited this README locally: **stage, commit, and push yourself** (only your account should appear on commits).
