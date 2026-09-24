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

# ⚡ DRISTI: Compound AI Routing Engine

**Dristi** is an ultra-fast, strictly-typed, non-autoregressive probabilistic decision engine. Built as a "System One" router, it handles routine AI routing tasks in milliseconds for fractions of a cent, intelligently routing difficult or uncertain queries to heavy "System 2" models.

Unlike standard generative models that hallucinate and take seconds to return text, Dristi returns strict JSON probabilities and logic routing decisions.

## 🏗️ Architecture: The Dual-Engine Cascade

This repository represents one half of the **Dristi Dual-Engine Cascade**. Dristi uses two specialized mathematical models to balance extreme speed and deep intelligence:

1. **V1 (The Speed Demon):** Built on `DistilBERT`. Hits ~40ms latency and runs on GPU. The ultimate frontline router.
2. **V2 (The Heavy Thinker):** Built on `DeBERTa-v3`. Uses Disentangled Attention to understand complex sarcasm and highly technical logic. Loaded into CPU RAM as a local safety net.

When integrated using the Dristi framework, queries flow to **V1** first. If V1 lacks confidence or mathematically detects an anomaly, it instantly cascades the query to **V2** entirely locally.

## ✨ Enterprise Features
* **Mathematical Self-Awareness (OOD Detection):** Rejects alien or unsafe queries using Cosine Similarity against a pre-computed mathematical centroid.
* **Compound AI Routing:** Natively cascades from Fast models to Heavy models (or external APIs) based on calibrated confidence thresholds.
* **Calibrated Confidence:** Uses L-BFGS Temperature Scaling.
* **Self-Healing API:** Designed to actively learn from human/API feedback in production.

## 🚀 How to Use

To use these weights, you should use the official open-source Dristi inference engine, which handles the complex routing, OOD detection, and FastAPI deployment automatically.

1. Clone the core architecture:
   ```bash
   git clone https://github.com/SWARAJCHATTARAJ/DRISTI.git
   cd DRISTI
   ```
2. Download these model weights (`pytorch_model.bin`) and place them in the `checkpoints/` directory of your cloned repository.
3. Start the dual-engine router or the FastAPI server!
   ```bash
   python -m inference.dual_engine
   # or
   uvicorn inference.api:app --host 0.0.0.0 --port 8000
   ```

*Note: The custom architecture python file (`dristi_model.py`) is included in this Hugging Face repository for reference and compatibility with advanced HF loaders, but the official Dristi pipeline is recommended for production deployment.*
