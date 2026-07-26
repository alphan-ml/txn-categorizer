"""Transaction categorizer: fast-path classifier with LLM fallback.

Architecture:
  request -> fast classifier (<10ms) -> confidence >= threshold? return
                                     -> else: LLM fallback -> label logged
                                        as training data for next retrain
"""
import time
import json
import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.fast_path import FastClassifier
from app.llm_fallback import llm_categorize
from app.metrics import METRICS

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
FEEDBACK_LOG = Path(os.getenv("FEEDBACK_LOG", "data/llm_labels.jsonl"))

app = FastAPI(title="txn-categorizer", version="0.1.0")
clf = FastClassifier.load_or_train()


class Txn(BaseModel):
    description: str
    amount: float | None = None


class Prediction(BaseModel):
    category: str
    confidence: float
    path: str  # "fast" | "llm"
    latency_ms: float


@app.post("/categorize", response_model=Prediction)
async def categorize(txn: Txn):
    t0 = time.perf_counter()
    category, confidence = clf.predict(txn.description)

    if confidence >= CONFIDENCE_THRESHOLD:
        latency = (time.perf_counter() - t0) * 1000
        METRICS.record(path="fast", latency_ms=latency, confidence=confidence)
        return Prediction(category=category, confidence=confidence,
                          path="fast", latency_ms=round(latency, 2))

    # Slow path: LLM fallback; its label becomes training data
    llm_category = await llm_categorize(txn.description)
    latency = (time.perf_counter() - t0) * 1000
    METRICS.record(path="llm", latency_ms=latency, confidence=confidence)

    FEEDBACK_LOG.parent.mkdir(exist_ok=True)
    with FEEDBACK_LOG.open("a") as f:
        f.write(json.dumps({"description": txn.description,
                            "label": llm_category,
                            "fast_path_guess": category,
                            "fast_path_confidence": confidence,
                            "ts": time.time()}) + "\n")

    return Prediction(category=llm_category, confidence=confidence,
                      path="llm", latency_ms=round(latency, 2))


@app.get("/metrics")
def metrics():
    """Fallback rate, latency percentiles — consumed by the monitoring page."""
    return METRICS.summary()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": clf.version}
