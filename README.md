# ML Ops Assignment 1 — g25ai1025

**Repository:** https://github.com/kst577/g25ai1025-mlops-assignment1

Boston Housing price prediction using classical ML models (Decision Tree & Kernel Ridge) with an automated GitHub Actions CI pipeline.

**Student:** g25ai1025  
**Reproducibility seed:** `1025` (derived from roll number)

---

## Branch Structure

| Branch | Purpose |
|--------|---------|
| `main` | Project documentation and merged codebase |
| `dtree` | Decision Tree regressor (`train.py`, `misc.py`, `requirements.txt`) |
| `kernelridge` | Kernel Ridge regressor (`train2.py`) + GitHub Actions CI |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kst577/g25ai1025-mlops-assignment1.git
cd g25ai1025-mlops-assignment1
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

> `requirements.txt` is added on the `dtree` branch and merged into `main`.

---

## How to Run

After the `dtree` branch is merged, train and evaluate the Decision Tree model:

```bash
python train.py
```

After the `kernelridge` branch is set up, train and evaluate the Kernel Ridge model:

```bash
python train2.py
```

Each script prints the **average MSE** on the test set.

---

## Project Layout

```
.
├── misc.py              # Shared data loading, preprocessing, training, evaluation
├── train.py             # DecisionTreeRegressor workflow
├── train2.py            # KernelRidge workflow
├── requirements.txt     # Python dependencies
├── .github/workflows/   # CI pipeline (kernelridge branch)
└── README.md
```

---

## Dataset

The Boston Housing dataset is loaded manually (deprecated in scikit-learn) via `misc.load_data()` using the official workaround from the [scikit-learn documentation](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html).
