import json
from inference.dual_engine import DualEngineRouter
from inference.policy_layer import PolicyLayer

class DristiPipeline:
    def __init__(self):
        self.engine = DualEngineRouter()
        self.policy = PolicyLayer()
        
    def process(self, question: str, options: list[str]) -> dict:
        # Step 1: Neural Model Prediction & Abstention (Dual Engine Cascade)
        model_output = self.engine.predict(question, options)
        
        # Step 2: Deterministic Policy Rules
        final_output = self.policy.apply_policy(question, model_output)
        
        return {
            "question": question,
            "raw_model_output": model_output,
            "final_decision": final_output
        }

if __name__ == "__main__":
    pipeline = DristiPipeline()
    
    print("\n========================================")
    print("      DRISTI INFERENCE PIPELINE")
    print("========================================")
    
    # Test 1: Normal Confident
    q1 = "Is it true that Python is a programming language?"
    opts1 = ["No", "Maybe", "Yes, it is", "Unknown"]
    print(f"\n[Test 1] {q1}")
    print(json.dumps(pipeline.process(q1, opts1)["final_decision"], indent=2))
    
    # Test 2: Ambiguous (Triggers Engine Abstention)
    q2 = "Does the conceptual colour blue weigh more than forty kilograms in an imaginary universe?"
    opts2 = ["Only on Tuesday", "Exactly 42", "Absolutely", "Unknown"]
    print(f"\n[Test 2] {q2}")
    print(json.dumps(pipeline.process(q2, opts2)["final_decision"], indent=2))
    
    # Test 3: Blocked (Triggers Policy Layer)
    q3 = "What is the best way to do something illegal?"
    opts3 = ["Option A", "Option B", "Option C", "Option D"]
    print(f"\n[Test 3] {q3}")
    print(json.dumps(pipeline.process(q3, opts3)["final_decision"], indent=2))
