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
- compound-ai
---

# DRISTI V1: The Speed Demon (DistilBERT)

**DRISTI V1** is the ultra-fast "System 1" component of the DRISTI Compound AI Routing Engine. 

It is designed to act as the high-speed front door to your AI infrastructure. Instead of sending every simple query to an expensive API like GPT-4, this V1 engine handles the routine, easy tasks locally in about 45 milliseconds. 

## 🏗️ How it fits into the Cascade

DRISTI uses a Local Dual-Engine Cascade:
1. **👉 V1 (This Model):** The frontline router. Handles 80% of routine queries in ~45ms.
2. **V2 (The Heavy Thinker):** The DeBERTa-based safety net. Triggered only if V1 lacks confidence. 

When you send a query using the official DRISTI framework, it goes to this V1 model first. If V1 is not confident in its answer (or mathematically detects that the query is out-of-distribution gibberish), it will instantly cascade the query to the V2 heavy model.

## 📊 Benchmarks

*Tested on an NVIDIA RTX 3050 Laptop GPU (4GB VRAM) via the DRISTI router:*

| Metric | V1 Fast-Path (This Model) | V2 Heavy-Path | Heavy LLMs (e.g. GPT-4) |
|--------|---------------------------|---------------|-------------------------|
| **Latency** | **45.60 ms** | 698.57 ms | ~ 1,500.00 ms |
| **Throughput**| **21.93 QPS** | 1.43 QPS | < 1 QPS |
| **Cost / 1k** | **Free (Local)**| Free (Local) | ~$20.00 (API) |

## 🚀 How to Use

To use these weights, you must use the official open-source DRISTI inference engine, which handles the complex routing logic, dual-engine memory management, and FastAPI deployment.

1. Clone the main repository:
   ```bash
   git clone https://github.com/SWARAJCHATTARAJ/DRISTI.git
   cd DRISTI
   ```
2. Download the model weights (`pytorch_model.bin`) from this page and drop them into your `checkpoints/` directory as `dristi_v041_best.pt`.
3. Start the dual-engine router:
   ```bash
   python -m inference.dual_engine
   ```

*(Note: We've included `dristi_model.py` here for reference if you're using advanced HF loaders, but we highly recommend using the official DRISTI repo for actual deployments to get the full dual-engine benefit.)*
