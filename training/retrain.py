"""Retrain the fast-path model on seed + accumulated LLM labels.

Run on a schedule (cron / GitHub Action). Each retrain absorbs the LLM's
labels, so the fast path handles more traffic and the fallback rate drops.
"""
import sys
sys.path.insert(0, ".")
from app.fast_path import FastClassifier

if __name__ == "__main__":
    clf = FastClassifier.train()
    print(f"retrained -> models/fast_path.joblib (version {clf.version})")
