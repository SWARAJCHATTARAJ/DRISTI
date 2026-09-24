import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from inference.config import MODEL_NAME, CHECKPOINT_PATH, CALIBRATION_PATH, MAX_LENGTH

class DristiEngine:
    def __init__(self, device=None, model_class=None, model_name=None, checkpoint_path=None, calibration_path=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use provided args or fallback to config defaults
        self.model_name = model_name or MODEL_NAME
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.calibration_path = calibration_path or CALIBRATION_PATH
        
        if model_class is None:
            from model.dristi_model_v03 import DristiModelV03
            model_class = DristiModelV03

        print(f"Loading Tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Loading Model: {model_class.__name__} from {self.checkpoint_path}...")
        self.model = model_class(model_name=self.model_name, num_scores=5)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        self.t_binary = 1.0
        self.t_choice = 1.0
        
        # Abstention Config
        self.min_confidence = 0.60
        self.min_margin = 0.02
        self.max_uncertainty = 0.40
        
        if os.path.exists(self.calibration_path):
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
                self.t_binary = calib.get("binary_temperature", 1.0)
                self.t_choice = calib.get("choice_temperature", 1.0)
                # Load calibrated thresholds if available, otherwise use defaults
                self.min_confidence = calib.get("min_confidence", self.min_confidence)
                self.min_margin = calib.get("min_margin", self.min_margin)
                self.max_uncertainty = calib.get("max_uncertainty", self.max_uncertainty)
            print(f"Loaded calibration: T_binary={self.t_binary:.4f}, T_choice={self.t_choice:.4f}")
        else:
            print("Warning: Calibration file not found. Using default temperatures (1.0).")
            
        self.ood_centroid = None
        ood_path = "checkpoints/ood_centroid.npy"
        if os.path.exists(ood_path):
            import numpy as np
            # Load the mathematical center of known training data
            self.ood_centroid = torch.tensor(np.load(ood_path)).to(self.device)
            self.min_ood_similarity = 0.15 # Lowered to prevent aggressive blocking of general queries
            print("Loaded OOD mathematical centroid for alien-query detection.")

    @torch.inference_mode()
    def predict(self, question: str, options: list[str]) -> dict:
        if not question or not options:
            raise ValueError("Question and options must be provided.")
        
        # 1. Tokenize Question
        q_tokens = self.tokenizer(
            question, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        q_input_ids = q_tokens["input_ids"].to(self.device)
        q_attention_mask = q_tokens["attention_mask"].to(self.device)

        # 2. Tokenize Options
        opt_texts = [question + " [SEP] " + opt for opt in options]
        o_tokens = self.tokenizer(
            opt_texts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        o_input_ids = o_tokens["input_ids"].unsqueeze(0).to(self.device)
        o_attention_mask = o_tokens["attention_mask"].unsqueeze(0).to(self.device)

        # 3. Model Forward
        q_out = self.model.forward_question(q_input_ids, q_attention_mask)
        o_logits = self.model.score_options(o_input_ids, o_attention_mask)

        # 4. Binary Decision (with Temperature Scaling)
        scaled_binary_logits = q_out["binary"] / self.t_binary
        binary_probs = F.softmax(scaled_binary_logits, dim=1)[0]
        prob_yes = float(binary_probs[1].item())
        prob_no = float(binary_probs[0].item())
        
        decision = "YES" if prob_yes >= prob_no else "NO"
        confidence = max(prob_yes, prob_no)
        uncertainty = 1.0 - confidence
        
        # 5. Choice Selection (with Temperature Scaling)
        scaled_choice_logits = o_logits / self.t_choice
        choice_probs = F.softmax(scaled_choice_logits, dim=1)[0]
        
        # Sort probabilities to find margin
        sorted_choice_probs, sorted_indices = torch.sort(choice_probs, descending=True)
        top_prob = float(sorted_choice_probs[0].item())
        runner_up_prob = float(sorted_choice_probs[1].item()) if len(sorted_choice_probs) > 1 else 0.0
        choice_margin = top_prob - runner_up_prob
        
        selected_idx = int(sorted_indices[0].item())
        selected_option = options[selected_idx]
        
        # 6. Ordinal Score
        threshold_probs = torch.sigmoid(q_out["ordinal"])
        expected_score = float(1.0 + threshold_probs.sum().item())
        final_score = max(1, min(5, int(round(expected_score))))

        # 7. Abstention Logic
        abstain = False
        ood_flag = False
        
        # Mathematical OOD Check
        if self.ood_centroid is not None:
            # q_out["state"] is the [CLS] embedding from DristiModelV03
            query_embedding = q_out["state"]
            similarity = F.cosine_similarity(query_embedding, self.ood_centroid.unsqueeze(0), dim=1).item()
            if similarity < self.min_ood_similarity:
                abstain = True
                ood_flag = True
                
        if confidence < self.min_confidence:
            abstain = True
        elif uncertainty > self.max_uncertainty:
            abstain = True
        elif choice_margin < self.min_margin:
            abstain = True

        if abstain:
            decision = "UNKNOWN"
            # We preserve the underlying stats so the caller knows WHY it abstained

        result = {
            "decision": decision,
            "confidence": round(confidence, 4),
            "score": final_score,
            "uncertainty": round(uncertainty, 4),
            "abstain": abstain,
            "selected_option": selected_option,
            "probabilities": [round(p, 4) for p in choice_probs.tolist()]
        }
        
        if self.ood_centroid is not None:
            result["ood_similarity"] = round(similarity, 4)
            result["ood_detected"] = ood_flag
            
        return result

if __name__ == "__main__":
    engine = DristiEngine()
    
    test_q_confident = "Is it true that Python is a programming language?"
    test_opts = ["No, it's a snake", "Maybe", "Yes, it is", "Unknown"]
    
    print("\n--- Test 1: Confident Prediction ---")
    print(json.dumps(engine.predict(test_q_confident, test_opts), indent=2))
    
    print("\n--- Test 2: Ambiguous Prediction (Should Abstain) ---")
    # Using a purely nonsensical question to guarantee binary uncertainty
    test_q_ambiguous = "Does the conceptual colour blue weigh more than forty kilograms in an imaginary universe?"
    test_opts_ambiguous = ["Only on Tuesday", "Exactly 42", "Absolutely", "Unknown"]
    print(json.dumps(engine.predict(test_q_ambiguous, test_opts_ambiguous), indent=2))
