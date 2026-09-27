---
license: mit
language:
- en
pipeline_tag: text-classification
tags:
- dristi
- router
- compound-ai
- system-one
- system-two
- classification
- fast-inference
---

# DRISTI: Compound AI Routing Engine

**DRISTI** is a high-speed routing engine designed to sit in front of your AI infrastructure. Instead of making slow, expensive API calls to heavy models for every single user request, DRISTI acts as a "System One" filter. It handles the routine, easy tasks locally in milliseconds, and only routes the tricky queries to heavier models when it lacks confidence.

And unlike generative models, it returns strict, predictable JSON instead of unstructured text.

## The Dual-Engine Cascade

This repository contains the weights for the **DRISTI Dual-Engine Cascade**. DRISTI uses two separate models working together to balance speed and intelligence:

1. **V1 (The Speed Demon):** A DistilBERT-based model that runs on your GPU. It hits ~40ms latency and handles the vast majority of requests.
2. **V2 (The Heavy Thinker):** A DeBERTa-v3 model loaded into CPU RAM. This acts as the local safety net to catch complex, technical, or nuanced queries that V1 isn't sure about.

When you send a query, it goes to V1 first. If V1 isn't confident (or detects the query is complete gibberish), it instantly cascades to V2 entirely locally.

## Key Features
* **Out-of-Distribution (OOD) Detection:** Safely rejects weird or unsafe queries by comparing them against what it was trained on.
* **Compound AI Routing:** Seamlessly hands off difficult queries from the fast model to the heavy model.
* **Calibrated Confidence:** Confidence scores actually map to real-world probability.
* **Self-Healing:** Built to ingest corrections and learn from human or API feedback in production.

## Benchmarks

*Tested on an NVIDIA RTX 3050 Laptop GPU (4GB VRAM)*

| Metric | V1 Fast-Path | V2 Heavy-Path | Heavy LLMs (e.g. GPT-4) |
|--------|--------------|---------------|-------------------------|
| **Latency** | 45.60 ms | 698.57 ms | ~ 1,500.00 ms |
| **Throughput**| 21.93 QPS | 1.43 QPS | < 1 QPS |
| **Cost / 1k** | Free (Local)| Free (Local) | ~$20.00 (API) |

## How to Use

To actually run these weights, you'll want to use the official DRISTI inference engine, which handles all the routing logic and the FastAPI server.

1. Clone the main repository:
   ```bash
   git clone https://github.com/SWARAJCHATTARAJ/DRISTI.git
   cd DRISTI
   ```
2. Download the model weights (`pytorch_model.bin`) from this page and drop them into the `checkpoints/` directory of your cloned repo.
3. Fire up the router or the API:
   ```bash
   python -m inference.dual_engine
   # or
   uvicorn inference.api:app --host 0.0.0.0 --port 8000
   ```

*(Note: We've included `dristi_model.py` here for reference if you're using advanced HF loaders, but we highly recommend using the official DRISTI repo for actual deployments.)*
