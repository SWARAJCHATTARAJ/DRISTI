# ⚡ DRISTI: The System-One AI Routing Engine

Dristi is an ultra-fast, strictly-typed, non-autoregressive probabilistic decision engine. Built as a "System One" router, Dristi acts as the high-speed front door to your AI infrastructure—handling 80% of routine tasks in milliseconds for fractions of a cent, and intelligently routing the difficult 20% to heavy "System 2" models.

Unlike standard generative models that hallucinate and take seconds to return text, Dristi returns strict JSON probabilities in **under 45 milliseconds**.

## ✨ Enterprise Features

- **⚡ Blazing Fast Inference:** Evaluates text, scores options, and makes decisions in ~40ms on standard hardware.
- **🛡️ Mathematical Self-Awareness (OOD Detection):** Dristi calculates the Cosine Similarity of incoming requests against a mathematical centroid of its known training data. If a query is "Out of Distribution" (alien or gibberish), it safely abstains.
- **🧠 Local Dual-Engine Cascade:** Dristi natively loads both a fast System 1 (DistilBERT) and a heavy System 2 (DeBERTa) model into memory. If System 1 lacks confidence, it cascades entirely locally without needing an external API!
- **🔄 Self-Healing API (Active Learning):** Built-in FastAPI endpoint (`/feedback`) automatically ingests human or System 2 corrections and permanently appends them to the real-world dataset. Dristi gets smarter every day, completely on autopilot.
- **⚖️ Calibrated Probabilities:** Uses L-BFGS Temperature Scaling to ensure that a "90% confidence" score mathematically correlates to a 90% chance of being correct.

## 🏗️ Dual Architecture (V1 vs V2)

Dristi ships with two distinct mathematical models connected via the `inference.dual_engine` router:

*   **Version 1 (The Speed Demon):** Built on `DistilBERT`. Hits ~40ms latency and runs on GPU. The ultimate frontline router.
*   **Version 2 (The Heavy Thinker):** Built on `DeBERTa-v3`. Uses Disentangled Attention to understand complex sarcasm and highly technical logic. Loaded into CPU RAM by default to save VRAM, acting as the local safety net.

## 📊 Benchmarks

*Hardware: NVIDIA RTX 3050 Laptop GPU (4GB VRAM)*

| Metric | System 1 (V1 Fast-Path) | System 2 (V2 Heavy-Path) | Heavy LLM (e.g. GPT-4) |
|--------|-------------------------|--------------------------|------------------------|
| **Avg Latency** | **42.39 ms** | ~ 150.00 ms | ~ 1,500.00 ms |
| **Throughput** | **23.59 QPS** | ~ 5 QPS | < 1 QPS |
| **Cost per 1k** | **$0.00 (Local)** | **$0.00 (Local)** | ~$20.00 (API) |

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/SWARAJCHATTARAJ/DRISTI.git
cd DRISTI
# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Test the Local Dual Engine
Run the local inference script to see the cascade router intelligently switch between the Fast and Heavy models:
```bash
python -m inference.dual_engine
```

### 3. Start the Self-Healing API
Run the FastAPI server to expose the Dual Engine and the `/feedback` loop to your web apps.
```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

### 4. Make a Routing Request
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Is Python a programming language?",
  "options": ["Yes", "No", "Maybe"]
}'
```

**Instant JSON Response:**
```json
{
  "decision": "YES",
  "confidence": 0.9509,
  "score": 5,
  "uncertainty": 0.0491,
  "abstain": false,
  "selected_option": "Yes",
  "probabilities": [0.95, 0.03, 0.02],
  "policy_flags": [],
  "ood_similarity": 0.852,
  "ood_detected": false,
  "engine_used": "V1_FAST (DistilBERT)"
}
```

## 📈 Training & Exporting

### Training the Heavy Thinker
To train the DeBERTa-v3 model on your specific domain data:
```bash
python -m training.train_dristi_v06_deberta
```

### Exporting to Hugging Face
Once trained, package both your V1 and V2 models instantly for Hugging Face:
```bash
python export_model.py
```
This will create an `hf_export` directory containing standard `transformers` repositories ready to upload!

## 🤝 Open Source (MIT License)
Dristi is completely open-source. The architecture proves that you don't need a $40M budget or a proprietary API to build a world-class, mathematically safe System One engine.
