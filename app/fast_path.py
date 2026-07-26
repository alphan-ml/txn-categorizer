"""Fast-path classifier. TF-IDF char n-grams + logistic regression.

Char n-grams because transaction strings are messy ("SQ *BLUE BOTTLE 4421",
"AMZN Mktp US*2K4"), and character-level features handle merchant-code noise
far better than word tokens. Target: <10ms inference on CPU.
"""
import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/fast_path.joblib")
SEED_DATA = Path("data/seed.jsonl")
LLM_LABELS = Path("data/llm_labels.jsonl")


class FastClassifier:
    def __init__(self, pipeline: Pipeline, version: str):
        self.pipeline = pipeline
        self.version = version

    def predict(self, description: str) -> tuple[str, float]:
        proba = self.pipeline.predict_proba([description])[0]
        idx = proba.argmax()
        return self.pipeline.classes_[idx], float(proba[idx])

    @classmethod
    def load_or_train(cls) -> "FastClassifier":
        if MODEL_PATH.exists():
            bundle = joblib.load(MODEL_PATH)
            return cls(bundle["pipeline"], bundle["version"])
        return cls.train()

    @classmethod
    def train(cls) -> "FastClassifier":
        """Train on seed data + any accumulated LLM feedback labels."""
        texts, labels = [], []
        for path in (SEED_DATA, LLM_LABELS):
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                texts.append(row["description"])
                labels.append(row["label"])

        if not texts:
            raise RuntimeError(f"No training data. Expected {SEED_DATA}.")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                      min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=10.0)),
        ])
        pipeline.fit(texts, labels)

        version = time.strftime("%Y%m%d-%H%M%S")
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"pipeline": pipeline, "version": version}, MODEL_PATH)
        return cls(pipeline, version)
