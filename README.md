# DRISTI: The System-One Routing Engine

Most AI models are slow, expensive, and unpredictable. They hallucinate, take several seconds to stream a response, and cost a fortune at scale. 

**DRISTI** is built to solve that. It acts as the high-speed "System One" front door for your AI infrastructure. Instead of sending every simple query to an expensive API like GPT-4, DRISTI handles the routine, easy tasks locally in about 40 milliseconds. It only routes the difficult questions to the heavy models when it really needs to.

And it doesn't generate text—it returns strict, predictable JSON.

## Why use DRISTI?

- **It's insanely fast:** We're talking ~40ms on standard hardware for most queries. 
- **It knows when it's confused (OOD Detection):** If you send it gibberish or something completely outside its training, it calculates the mathematical distance from what it knows and safely abstains rather than guessing.
- **Local Dual-Engine Cascade:** This is the core feature. DRISTI loads two models into memory: a lightning-fast one (DistilBERT) and a heavier, smarter one (DeBERTa). If the fast model isn't confident, DRISTI automatically cascades the query to the heavy model locally. No API calls required.
- **It gets smarter on autopilot:** There's a built-in FastAPI endpoint that lets you feed corrections back into the dataset, continuously improving the model in production.
- **Trustworthy confidence scores:** A 90% confidence score actually means it has a 90% chance of being right, thanks to built-in temperature scaling.

## The Two Engines (V1 vs V2)

DRISTI ships with two models working together:

* **V1 (The Speed Demon):** Built on DistilBERT. This runs on your GPU, hits ~40ms, and handles the bulk of the work.
* **V2 (The Heavy Thinker):** Built on DeBERTa-v3. This one sits in your CPU RAM (to save VRAM) and catches the complex, nuanced queries that V1 isn't sure about.

## Benchmarks

*Tested on an NVIDIA RTX 3050 Laptop GPU (4GB VRAM)*

| Metric | V1 Fast-Path | V2 Heavy-Path | Heavy LLMs (e.g. GPT-4) |
|--------|--------------|---------------|-------------------------|
| **Latency** | 45.60 ms | 698.57 ms | ~ 1,500.00 ms |
| **Throughput**| 21.93 QPS | 1.43 QPS | < 1 QPS |
| **Cost / 1k** | Free (Local)| Free (Local) | ~$20.00 (API) |

## Quickstart

### 1. Get the code
```bash
git clone https://github.com/SWARAJCHATTARAJ/DRISTI.git
cd DRISTI
.\.venv\Scripts\Activate.ps1
```

### 2. Test the router locally
Want to see the dual-engine switch between the fast and heavy models in real time?
```bash
python -m inference.dual_engine
```

### 3. Run the benchmarks
You can benchmark both engines on your own hardware to see the speed difference:
```bash
python -m inference.benchmark
```

### 4. Fire up the API
Start the FastAPI server so your web apps can start talking to the router.
```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

### 5. Send a request
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Is Python a programming language?",
  "options": ["Yes", "No", "Maybe"]
}'
```

You'll get an instant JSON response back looking something like this:
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

## Training & Exporting

### Train the Heavy Thinker
If you want to fine-tune the DeBERTa-v3 model on your own data:
```bash
python -m training.train_dristi_v06_deberta
```

### Export to Hugging Face
Once you're happy with your models, you can package both V1 and V2 for Hugging Face in one go:
```bash
python export_model.py
```
This drops everything into an `hf_export` directory, ready to be uploaded.

## Open Source
DRISTI is MIT licensed. It's built to prove you don't need massive budgets or expensive APIs to build a fast, reliable AI routing layer.
