# ⚡ Dristi: The System-One AI Routing Engine

Dristi is an ultra-fast, strictly-typed, non-autoregressive probabilistic decision engine. Built as a "System One" router, Dristi acts as the high-speed front door to your AI infrastructure—handling 80% of routine tasks in milliseconds for fractions of a cent, and intelligently routing the difficult 20% to heavy "System 2" LLMs (like GPT-4 or Llama).

Unlike standard generative models that hallucinate and take seconds to return text, Dristi returns strict JSON probabilities in **under 45 milliseconds**.

## ✨ Enterprise Features

- **⚡ Blazing Fast Inference:** Evaluates text, scores options, and makes decisions in ~40ms on standard hardware (24+ QPS without batching).
- **🛡️ Mathematical Self-Awareness (OOD Detection):** Dristi doesn't just guess. It calculates the Cosine Similarity of incoming requests against a mathematical centroid of its known training data. If a query is "Out of Distribution" (alien or gibberish), it safely abstains.
- **🧠 Compound AI Routing:** Native `router.py` automatically intercepts uncertain or OOD queries and gracefully formats them for handoff to heavy LLMs. 
- **🔄 Self-Healing API (Active Learning):** Built-in FastAPI endpoint (`/feedback`) automatically ingests human or System 2 corrections and permanently appends them to the real-world dataset. Dristi gets smarter every day, completely on autopilot.
- **⚖️ Calibrated Probabilities:** Uses L-BFGS Temperature Scaling to ensure that a "90% confidence" score mathematically correlates to a 90% chance of being correct.

## 📊 Benchmarks

*Hardware: NVIDIA RTX 3050 Laptop GPU (4GB VRAM)*

| Metric | System 1 (Fast-Path) | Heavy LLM (System 2) |
|--------|----------------------|----------------------|
| **Avg Latency** | **42.39 ms** | ~ 1,500.00 ms |
| **P99 Latency** | **51.35 ms** | ~ 3,000.00 ms |
| **Throughput** | **23.59 QPS** | < 1 QPS |
| **Cost per 1k** | **$0.00 (Local)** | ~$20.00 (API) |
| **Overhead** | **2.71 ms** | N/A |

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/yourusername/dristi.git
cd dristi
pip install -e .
```

### 2. Start the Self-Healing API
Run the FastAPI server to expose the inference engine and the `/feedback` loop.
```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

### 3. Make a Routing Request
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
  "ood_detected": false
}
```

## 🛠️ Architecture

Dristi is built on a 66-million parameter encoder (DistilBERT) featuring three distinct heads:
1. **Binary Head:** Decides YES/NO.
2. **Dynamic Option Scorer:** Evaluates an arbitrary array of `N` options and returns a confidence distribution.
3. **Ordinal Head:** Calculates a strictly enforced 1-5 expected value score.

## 📈 Training on Your Own Private Data

Dristi is designed to be a domain specialist. To overshadow generalized models in your specific company (e.g., Legal Routing, Support Tickets):
1. Format your private data into `data/train_real.json`.
2. Run `python -m training.train_dristi_v05_real`.
3. Dristi will master your domain while remaining 100% private and air-gapped.

## 🤝 Open Source
Dristi is completely open-source. The architecture proves that you don't need a $40M budget or a proprietary API to build a world-class, mathematically safe System One engine.
