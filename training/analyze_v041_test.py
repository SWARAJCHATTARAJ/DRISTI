import json
import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax
from collections import Counter

from model.dristi_model_v03 import DristiModelV03

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"
TEST_FILE = "data/test_v041.json"
CHECKPOINT = "checkpoints/dristi_v041_best.pt"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODEL & DATA
# ============================================================

with open(TEST_FILE, "r", encoding="utf-8") as file:
    test_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = DristiModelV03(model_name=MODEL_NAME, num_scores=5)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ============================================================
# EVALUATION & ERROR COLLECTION
# ============================================================

wrong_examples = []
confidently_wrong_binary = []
uncertain_but_correct_binary = []

binary_actual_dist = Counter()
binary_pred_dist = Counter()
choice_actual_dist = Counter()
choice_pred_dist = Counter()
score_actual_dist = Counter()
score_pred_dist = Counter()

with torch.inference_mode():
    for index, item in enumerate(test_data):
        question = item["question"]
        options = item["options"]
        actual_binary = int(item["yes"])
        actual_choice = int(item["choice"])
        actual_score = int(item["score"])

        # Tokenization
        question_tokens = tokenizer(
            question, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        option_texts = [question + " [SEP] " + option for option in options]
        option_tokens = tokenizer(
            option_texts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )

        question_input_ids = question_tokens["input_ids"].to(device)
        question_attention_mask = question_tokens["attention_mask"].to(device)
        option_input_ids = option_tokens["input_ids"].unsqueeze(0).to(device)
        option_attention_mask = option_tokens["attention_mask"].unsqueeze(0).to(device)

        # Forward passes
        question_output = model.forward_question(question_input_ids, question_attention_mask)
        option_logits = model.score_options(option_input_ids, option_attention_mask)

        # Binary
        binary_probs = softmax(question_output["binary"], dim=1)[0]
        prob_yes = float(binary_probs[1].item())
        pred_binary = int(torch.argmax(question_output["binary"], dim=1).item())
        max_binary_prob = max(float(binary_probs[0]), prob_yes)

        # Choice
        choice_probs = softmax(option_logits, dim=1)[0]
        pred_choice = int(torch.argmax(option_logits, dim=1).item())

        # Score
        threshold_probs = torch.sigmoid(question_output["ordinal"])
        expected_score = float(1.0 + threshold_probs.sum().item())
        pred_score = max(1, min(5, int(round(expected_score))))
        
        # Tracking distributions for bias check
        binary_actual_dist[actual_binary] += 1
        binary_pred_dist[pred_binary] += 1
        choice_actual_dist[actual_choice] += 1
        choice_pred_dist[pred_choice] += 1
        score_actual_dist[actual_score] += 1
        score_pred_dist[pred_score] += 1

        # Error checks
        is_binary_wrong = pred_binary != actual_binary
        is_choice_wrong = pred_choice != actual_choice
        is_score_wrong = pred_score != actual_score
        
        if is_binary_wrong and max_binary_prob > 0.8:
            confidently_wrong_binary.append(index)
        if not is_binary_wrong and max_binary_prob < 0.6:
            uncertain_but_correct_binary.append(index)

        if is_binary_wrong or is_choice_wrong or is_score_wrong:
            wrong_examples.append({
                "index": index,
                "question": question,
                "options": options,
                "actual_binary": actual_binary,
                "pred_binary": pred_binary,
                "prob_yes": prob_yes,
                "actual_choice": actual_choice,
                "pred_choice": pred_choice,
                "choice_probs": choice_probs.tolist(),
                "actual_score": actual_score,
                "pred_score": pred_score,
                "expected_score": expected_score,
                "is_binary_wrong": is_binary_wrong,
                "is_choice_wrong": is_choice_wrong,
                "is_score_wrong": is_score_wrong
            })


# ============================================================
# PRINT ERROR ANALYSIS
# ============================================================
print("=" * 70)
print("              DRISTI v0.4.1 ERROR ANALYSIS")
print("=" * 70)

print(f"\nTotal Errors Found: {len(wrong_examples)} out of {len(test_data)}")

print("\n--- BIAS & DISTRIBUTION CHECKS ---")
print("Binary Actual Dist :", dict(sorted(binary_actual_dist.items())))
print("Binary Pred Dist   :", dict(sorted(binary_pred_dist.items())))
print("Choice Actual Dist :", dict(sorted(choice_actual_dist.items())))
print("Choice Pred Dist   :", dict(sorted(choice_pred_dist.items())))
print("Score Actual Dist  :", dict(sorted(score_actual_dist.items())))
print("Score Pred Dist    :", dict(sorted(score_pred_dist.items())))

print("\n--- UNCERTAINTY FAILURES ---")
print(f"Confidently Wrong (Binary P > 0.8): {len(confidently_wrong_binary)} examples")
print(f"Uncertain but Correct (Binary P < 0.6): {len(uncertain_but_correct_binary)} examples")

print("\n" + "=" * 70)
print("ERROR DETAILS")
print("=" * 70)

for err in wrong_examples:
    print(f"\n[Test Index: {err['index']}]")
    print(f"Q: {err['question']}")
    for i, opt in enumerate(err['options']):
        print(f"   Option {i}: {opt}")
    
    print(f"\nBinary : Actual={err['actual_binary']}, Predicted={err['pred_binary']} (P_yes={err['prob_yes']:.4f}) {'[WRONG]' if err['is_binary_wrong'] else ''}")
    print(f"Choice : Actual={err['actual_choice']}, Predicted={err['pred_choice']} {'[WRONG]' if err['is_choice_wrong'] else ''}")
    for i, p in enumerate(err['choice_probs']):
        print(f"         P(Opt {i}) = {p:.4f}")
    
    print(f"Score  : Actual={err['actual_score']}, Predicted={err['pred_score']} (Expected={err['expected_score']:.4f}) {'[WRONG]' if err['is_score_wrong'] else ''}")
    print("-" * 50)

print("\n" + "=" * 70)
print("DRISTI v0.4.1 ERROR ANALYSIS COMPLETE")
print("=" * 70)
