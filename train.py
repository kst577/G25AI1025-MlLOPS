#!/usr/bin/env python3
"""
Decision Tree regression experiment — ML Ops Assignment 1 — g25ai1025.

All steps go through helpers in misc.py so other regressors reuse the same code path.
"""

from sklearn.tree import DecisionTreeRegressor

from misc import run_experiment


def main():
    model = DecisionTreeRegressor(random_state=1025)
    run_experiment(model, "DecisionTreeRegressor", scale_features=False)


if __name__ == "__main__":
    main()
