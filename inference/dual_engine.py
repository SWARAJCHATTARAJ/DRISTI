import os
import torch
from inference.dristi_engine import DristiEngine
from model.dristi_model_v03 import DristiModelV03
from model.dristi_model_v04 import DristiModelV04 # Deberta based model

class DualEngineRouter:
    def __init__(self):
        print("==================================================")
        print("          INITIALIZING DUAL ENGINE ROUTER         ")
        print("==================================================")
        
        # 1. Load the Speed Demon (DistilBERT)
        print("\n[Dual Engine] Initializing System 1 (Speed Demon)...")
        self.fast_engine = DristiEngine(
            model_class=DristiModelV03,
            model_name="distilbert-base-uncased",
            checkpoint_path="checkpoints/dristi_v041_best.pt",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 2. Load the Heavy Thinker (Smart Fallback Logic)
        print("\n[Dual Engine] Initializing System 2 (Heavy Thinker)...")
        
        v2_deberta_path = "checkpoints/dristi_v06_deberta_best.pt"
        v2_distil_fallback = "checkpoints/dristi_v05_real_best.pt"

        if os.path.exists(v2_deberta_path):
            print(" -> Found DeBERTa checkpoint! Loading true Heavy Thinker...")
            self.heavy_engine = DristiEngine(
                model_class=DristiModelV04,
                model_name="microsoft/deberta-v3-small", 
                checkpoint_path=v2_deberta_path,
                device=torch.device("cpu") # CPU to save VRAM on 4GB GPU
            )
            self.v2_name = "V2_HEAVY (DeBERTa-v3)"
        else:
            print(" -> DeBERTa checkpoint not found. Falling back to DistilBERT V05 as System 2...")
            self.heavy_engine = DristiEngine(
                model_class=DristiModelV03,
                model_name="distilbert-base-uncased", 
                checkpoint_path=v2_distil_fallback,
                device=torch.device("cpu")
            )
            self.v2_name = "V2_FALLBACK (DistilBERT)"
        
        # Routing Thresholds
        self.min_confidence = 0.85
        print("\n[Dual Engine] Initialization Complete.")

    def predict(self, question: str, options: list[str]) -> dict:
        print(f"\n[Dual Engine] Incoming query: '{question}'")
        
        # Phase 1: Try Fast Engine (~40ms)
        fast_result = self.fast_engine.predict(question, options)
        
        # If System 1 is confident and didn't abstain, use it directly!
        if not fast_result.get("abstain") and fast_result.get("confidence", 0) >= self.min_confidence:
            fast_result["engine_used"] = "V1_FAST (DistilBERT)"
            print(f"[Dual Engine] System 1 succeeded with {fast_result['confidence']:.2f} confidence.")
            return fast_result
            
        # Phase 2: Fast Engine is uncertain, cascade to Heavy Engine (~150ms)
        print(f"[Dual Engine] System 1 lacked confidence ({fast_result.get('confidence', 0):.2f}). Cascading to System 2...")
        heavy_result = self.heavy_engine.predict(question, options)
        heavy_result["engine_used"] = self.v2_name
        
        return heavy_result

if __name__ == "__main__":
    import json
    router = DualEngineRouter()
    
    print("\n--- Test 1: Easy Query (Should use System 1) ---")
    q1 = "Is Python a programming language?"
    opts1 = ["No", "Yes", "Maybe"]
    print(json.dumps(router.predict(q1, opts1), indent=2))
    
    print("\n--- Test 2: Hard Query (Should cascade to System 2) ---")
    q2 = "Does the conceptual colour blue weigh more than forty kilograms in an imaginary universe?"
    opts2 = ["Only on Tuesday", "Exactly 42", "Absolutely", "Unknown"]
    print(json.dumps(router.predict(q2, opts2), indent=2))
