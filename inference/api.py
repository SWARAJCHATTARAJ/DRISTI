from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from inference.pipeline import DristiPipeline

app = FastAPI(title="Dristi Decision Engine API", version="0.4.1")

# Initialize pipeline on startup
pipeline = DristiPipeline()

class PredictRequest(BaseModel):
    question: str
    options: List[str]

class PredictResponse(BaseModel):
    decision: str
    confidence: float
    score: int
    uncertainty: float
    abstain: bool
    selected_option: str
    probabilities: List[float]
    policy_flags: List[str]
    ood_similarity: float = None
    ood_detected: bool = False
    engine_used: str = None

class FeedbackRequest(BaseModel):
    question: str
    options: List[str]
    correct_choice_idx: int
    correct_score_1_to_5: int

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "Dristi-v0.4.1"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Empty question provided.")
    if not request.options or len(request.options) < 2:
        raise HTTPException(status_code=400, detail="At least 2 options required.")
        
    try:
        result = pipeline.process(request.question, request.options)
        return result["final_decision"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_batch(requests: List[PredictRequest]):
    results = []
    for req in requests:
        if not req.question or not req.options or len(req.options) < 2:
            raise HTTPException(status_code=400, detail="Invalid question or options in batch.")
            
        try:
            res = pipeline.process(req.question, req.options)
            results.append(res["final_decision"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return results

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    ACTIVE LEARNING LOOP:
    When a human (or System 2 LLM) correctly answers a question that Dristi abstained on,
    this endpoint saves the correct answer back into the training dataset.
    """
    import os
    import json
    
    train_path = "data/train_real.json"
    if not os.path.exists(train_path):
        raise HTTPException(status_code=500, detail="Training dataset not found.")
        
    # Read existing data
    with open(train_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # Format the new real-world data point
    new_entry = {
        "question": feedback.question,
        "options": feedback.options,
        "yes": 1, # Default positive for demo
        "choice": feedback.correct_choice_idx,
        "score": feedback.correct_score_1_to_5
    }
    
    # Append and save (Self-Healing Loop)
    dataset.append(new_entry)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    return {
        "status": "success", 
        "message": "Feedback saved! Dristi will learn from this in the next training cycle.",
        "dataset_size": len(dataset)
    }

if __name__ == "__main__":
    # Local quick-test using FastAPI TestClient to avoid blocking the terminal
    from fastapi.testclient import TestClient
    import json
    
    print("\nStarting local API test client...")
    client = TestClient(app)
    
    print("\n--- Testing GET /health ---")
    response = client.get("/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    print("\n--- Testing POST /predict ---")
    payload = {
        "question": "Is FastAPI used for building web APIs?",
        "options": ["No", "Maybe", "Yes", "Unknown"]
    }
    response = client.post("/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    print("\n--- Testing POST /predict (Malformed Input) ---")
    bad_payload = {
        "question": "",
        "options": ["Single Option"]
    }
    response = client.post("/predict", json=bad_payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    print("\n--- Testing POST /feedback (Self-Healing Loop) ---")
    feedback_payload = {
        "question": "What is the emotion of this text: 'I am so confused by PyTorch'?",
        "options": ["Sadness", "Joy", "Fear", "Surprise", "Anger", "Confusion"],
        "correct_choice_idx": 5,
        "correct_score_1_to_5": 4
    }
    response = client.post("/feedback", json=feedback_payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    print("\nAPI architecture is ready. You can serve this using 'uvicorn inference.api:app'")
