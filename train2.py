#!/usr/bin/env python3
"""
Kernel Ridge regression — ML Ops Assignment 1 — g25ai1025.

All steps go through helpers in misc.py (same path as train.py).
"""

from sklearn.kernel_ridge import KernelRidge

from misc import run_experiment


def main():
    # Explicit gamma keeps RBF KernelRidge well-conditioned on scaled Boston features.
    model = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.05)
    run_experiment(model, "KernelRidge", scale_features=True)


if __name__ == "__main__":
    main()
