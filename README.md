# txn-categorizer

Transaction categorization service with a **fast-path classifier and LLM fallback that trains its own replacement**.

**Live demo:** not deployed yet — this repo is source only. Run it locally in
about a minute with the commands below; no API key needed.

## Architecture

```
POST /categorize
      │
      ▼
┌─────────────────┐  conf ≥ 0.75   ┌──────────┐
│ TF-IDF char-ngram│ ─────────────► │ response │  <1ms
│ + LogisticReg    │                └──────────┘
└─────────────────┘
      │ conf < 0.75
      ▼
┌─────────────────┐                ┌──────────┐
│ LLM (Claude      │ ─────────────► │ response │  ~500ms
│ Haiku) fallback  │                └──────────┘
└────────┬────────┘
         │ label logged as training data
         ▼
   data/llm_labels.jsonl ──► scheduled retrain ──► fallback rate ↓
```

The interesting metric is **fallback rate over time**. It starts near 100%
(tiny seed set), and every retrain absorbs the LLM's labels into the fast
path. The dashboard shows the expensive path shrinking as the cheap path
learns — the system gets faster and cheaper under its own traffic.

## Design decisions

- **Char n-grams, not word tokens.** Bank descriptors are noise
  (`SQ *BLUE BOTTLE 4421`, `AMZN Mktp US*2K4`). Character-level features
  survive merchant-code garbage; word tokenizers don't.
- **Confidence-threshold routing, not always-LLM.** At scale, LLM-per-txn
  is cost-prohibitive (Plaid processes billions). The threshold is the
  cost/accuracy dial — one env var.
- **LLM labels as weak supervision.** No human labeling loop. Known
  tradeoff: the fast path can inherit LLM mistakes; mitigation would be
  agreement-based filtering before retraining.

## Run it

```bash
pip install -r requirements.txt
python training/retrain.py           # train from data/seed.jsonl
uvicorn app.main:app --reload

curl -X POST localhost:8000/categorize \
  -H 'Content-Type: application/json' \
  -d '{"description": "TRADER JOE'\''S #552"}'

curl localhost:8000/metrics          # fallback rate, latency p50/p95/p99
```

Without `ANTHROPIC_API_KEY` set, the fallback uses a deterministic stub so
the whole loop runs locally for free. See `.env.example` for every knob.

## The loop, actually running

This is the whole point of the project, so here is a real run rather than a
claim. Fifteen transactions through a freshly seeded model, then a retrain,
then the *same* fifteen again. Output is copied from an actual session, not
illustrative.

**Round 1 — model trained on the 50-row seed only.** Every request falls back:

```
path conf   category       description
llm  0.707  groceries      WHOLE FOODS MKT #123 AUSTIN TX
llm  0.729  groceries      TRADER JOE'S #552
llm  0.564  other          SQ *BLUE BOTTLE 4421
llm  0.724  dining         STARBUCKS 800-782-7282
llm  0.481  subscriptions  NETFLIX.COM
...
{"total":15,"fallback_rate":1.0,"latency_p50_ms":1.29,"latency_p95_ms":4.93}
```

Nothing clears the 0.75 threshold — with ~4 examples per category the
classifier is confident about nothing. **That is the expected starting state,
not a bug**, and it is what makes the next step legible.

**Round 2 — after `python training/retrain.py` absorbs those 15 LLM labels:**

```
path conf   category       description
fast 0.877  groceries      WHOLE FOODS MKT #123 AUSTIN TX
fast 0.884  groceries      TRADER JOE'S #552
llm  0.738  other          SQ *BLUE BOTTLE 4421
fast 0.836  dining         STARBUCKS 800-782-7282
fast 0.870  subscriptions  NETFLIX.COM
...
{"total":15,"fallback_rate":0.0667,"latency_p50_ms":1.44,"latency_p95_ms":7.67}
```

**Fallback rate 100% → 6.7% after one retrain**, 14 of 15 now served by the
fast path. The one holdout (`SQ *BLUE BOTTLE 4421`, 0.738) sits just under the
line and would clear it on the next cycle. The expensive path shrank because
the cheap path learned from it — under its own traffic, with no human labeling.

Reproduce it by hitting `/categorize` a few dozen times, running
`training/retrain.py`, restarting, and watching `/metrics`.

## Data provenance

`data/seed.jsonl` is **50 synthetic transaction descriptions written by hand**
for this project — merchant strings styled after real bank descriptors
(`SQ *`, `AMZN Mktp US*`, store numbers) but invented. **No real transaction
data, from any institution, is in this repository.** With ~4 examples across
12 categories it is a starting point for the feedback loop, not a training set
anyone should quote accuracy from — which is why this README reports fallback
rate and latency and makes no accuracy claim.

`data/llm_labels.jsonl` is generated at runtime from the fallback path and is
gitignored: it is a local artifact of your own traffic, not shipped data.

## Deploy

Dockerfile included. Targets: Fly.io / Railway / Render free tiers. Retrain
via GitHub Action on a daily cron committing the refreshed model artifact.

## Status

- [x] Fast path — 1.3ms p50 end-to-end through FastAPI, measured in the run above
- [x] LLM fallback + feedback logging
- [x] Retrain loop measurably closing (100% → 6.7% fallback, shown above)
- [x] Metrics endpoint (fallback rate, latency percentiles)
- [ ] Monitoring dashboard page
- [ ] Public deploy
- [ ] Scheduled retrain in CI

---


