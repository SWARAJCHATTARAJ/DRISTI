---
license: mit
language:
- en
pipeline_tag: text-classification
tags:
- dristi
- router
- system-two
- deberta
- compound-ai
---

# DRISTI V2: The Heavy Thinker (DeBERTa-v3)

**DRISTI V2** is the heavy-lifting "System 2" component of the DRISTI Compound AI Routing Engine. 

While the V1 engine (DistilBERT) acts as the ultra-fast frontline router, **V2 (DeBERTa-v3)** acts as the intelligent safety net. It is designed to be loaded into CPU RAM and is only triggered when V1 encounters a highly complex, nuanced, or deeply technical query that it isn't confident about.

## 🏗️ How it fits into the Cascade

DRISTI uses a Local Dual-Engine Cascade:
1. **V1 (The Speed Demon):** Handles 80% of routine queries in ~45ms.
2. **👉 V2 (This Model):** Handles the difficult 20% of queries in ~700ms. It uses DeBERTa's Disentangled Attention to understand complex sarcasm and tricky logic that smaller models miss.

When integrated using the official DRISTI framework, if V1 lacks confidence, the query is instantly cascaded locally to this V2 model without ever needing an external API call.

## 📊 Benchmarks

*Tested on an NVIDIA RTX 3050 Laptop GPU (4GB VRAM) via the DRISTI router:*

| Metric | V1 Fast-Path | V2 Heavy-Path (This Model) | Heavy LLMs (e.g. GPT-4) |
|--------|--------------|----------------------------|-------------------------|
| **Latency** | 45.60 ms | **698.57 ms** | ~ 1,500.00 ms |
| **Throughput**| 21.93 QPS | **1.43 QPS** | < 1 QPS |
| **Cost / 1k** | Free (Local)| **Free (Local)** | ~$20.00 (API) |

## 🚀 How to Use

To use these weights, you must use the official open-source DRISTI inference engine, which handles the complex routing logic, dual-engine memory management, and FastAPI deployment.

1. Clone the main repository:
   ```bash
   git clone https://github.com/SWARAJCHATTARAJ/DRISTI.git
   cd DRISTI
   ```
2. Download the model weights (`pytorch_model.bin`) from this page and drop them into your `checkpoints/` directory as `dristi_v06_deberta_best.pt`.
3. Start the dual-engine router:
   ```bash
   python -m inference.dual_engine
   ```

*(Note: We've included `dristi_model.py` here for reference if you're using advanced HF loaders, but we highly recommend using the official DRISTI repo for actual deployments to get the full dual-engine benefit.)*
