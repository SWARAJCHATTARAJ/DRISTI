---
license: mit
language:
- en
pipeline_tag: text-classification
tags:
- dristi
- router
- system-one
- distilbert
- classification
- active-learning
---

# ⚡ DRISTI-Router (V1 Speed Demon)

**Dristi** is an ultra-fast, strictly-typed, non-autoregressive probabilistic decision engine. Built as a "System One" router, it handles routine AI routing tasks in ~40 milliseconds, safely handing off uncertain or "Out-of-Distribution" (OOD) queries to heavy System 2 LLMs.

## Model Details
* **Architecture:** DistilBERT (Base, Uncased) with Custom Routing Heads (Binary, Choice, Ordinal)
* **Parameters:** ~66 Million
* **Latency:** ~40ms (RTX 3050)
* **License:** MIT

## Features
* **Mathematical Self-Awareness (OOD Detection):** Rejects alien or unsafe queries using Cosine Similarity.
* **Calibrated Confidence:** Uses L-BFGS Temperature Scaling.
* **Self-Healing API:** Designed to actively learn from human feedback in production.

## How to use
This model requires the custom `DristiModelV03` architecture from the main GitHub repository.

1. Clone the architecture: `git clone https://github.com/SWARAJCHATTARAJ/DRISTI.git`
2. Place this `pytorch_model.bin` and `calibration.json` into the `checkpoints/` folder.
3. Run the blazing-fast API!
