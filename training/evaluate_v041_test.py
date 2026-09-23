import json
import math
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.nn.functional import softmax

from model.dristi_model_v03 import DristiModelV03
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"
TEST_FILE = "data/test_v041.json"
CHECKPOINT = "checkpoints/dristi_v041_best.pt"
MAX_LENGTH = 128


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

with open(TEST_FILE, "r", encoding="utf-8") as file:
    test_data = json.load(file)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD MODEL
# ============================================================

model = DristiModelV03(
    model_name=MODEL_NAME,
    num_scores=5
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# RESULT STORAGE
# ============================================================

true_binary = []
pred_binary = []
binary_probabilities = []

true_choices = []
pred_choices = []
choice_probabilities = []

true_scores = []
pred_scores_expected = []
pred_scores_rounded = []


# ============================================================
# EVALUATION
# ============================================================

with torch.inference_mode():
    for item in test_data:
        question = item["question"]
        options = item["options"]
        actual_binary = int(item["yes"])
        actual_choice = int(item["choice"])
        actual_score = int(item["score"])

        # ----------------------------------------------------
        # Tokenize question
        # ----------------------------------------------------
        question_tokens = tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        question_input_ids = question_tokens["input_ids"].to(device)
        question_attention_mask = question_tokens["attention_mask"].to(device)

        # ----------------------------------------------------
        # Tokenize question + options
        # ----------------------------------------------------
        option_texts = [
            question + " [SEP] " + option
            for option in options
        ]
        option_tokens = tokenizer(
            option_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        option_input_ids = option_tokens["input_ids"].unsqueeze(0).to(device)
        option_attention_mask = option_tokens["attention_mask"].unsqueeze(0).to(device)

        # ----------------------------------------------------
        # QUESTION HEADS
        # ----------------------------------------------------
        question_output = model.forward_question(
            question_input_ids,
            question_attention_mask
        )
        binary_logits = question_output["binary"]
        ordinal_logits = question_output["ordinal"]

        # ----------------------------------------------------
        # BINARY PREDICTION
        # ----------------------------------------------------
        binary_probs = softmax(binary_logits, dim=1)
        probability_yes = float(binary_probs[0, 1].item())
        binary_prediction = int(torch.argmax(binary_logits, dim=1).item())

        # ----------------------------------------------------
        # CHOICE PREDICTION
        # ----------------------------------------------------
        option_logits = model.score_options(
            option_input_ids,
            option_attention_mask
        )
        choice_probs = softmax(option_logits, dim=1)
        choice_prediction = int(torch.argmax(choice_probs, dim=1).item())

        # ----------------------------------------------------
        # ORDINAL SCORE
        # ----------------------------------------------------
        threshold_probabilities = torch.sigmoid(ordinal_logits)
        expected_score = float(1.0 + threshold_probabilities.sum().item())
        rounded_score = int(round(expected_score))
        rounded_score = max(1, min(5, rounded_score))

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------
        true_binary.append(actual_binary)
        pred_binary.append(binary_prediction)
        binary_probabilities.append(probability_yes)

        true_choices.append(actual_choice)
        pred_choices.append(choice_prediction)
        choice_probabilities.append(choice_probs[0].cpu())

        true_scores.append(actual_score)
        pred_scores_expected.append(expected_score)
        pred_scores_rounded.append(rounded_score)


# ============================================================
# BINARY METRICS
# ============================================================
binary_accuracy = accuracy_score(true_binary, pred_binary)
binary_brier = sum((prob - actual) ** 2 for prob, actual in zip(binary_probabilities, true_binary)) / len(true_binary)

# Confusion matrix and derived metrics
binary_cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
yes_precision = precision_score(true_binary, pred_binary, pos_label=1, zero_division=0)
yes_recall = recall_score(true_binary, pred_binary, pos_label=1, zero_division=0)
no_precision = precision_score(true_binary, pred_binary, pos_label=0, zero_division=0)
no_recall = recall_score(true_binary, pred_binary, pos_label=0, zero_division=0)

# Baselines
majority_binary_class = max(set(true_binary), key=true_binary.count)
majority_binary_acc = sum(1 for x in true_binary if x == majority_binary_class) / len(true_binary)
constant_05_brier = sum((0.5 - actual) ** 2 for actual in true_binary) / len(true_binary)

# ============================================================
# CHOICE METRICS
# ============================================================
choice_accuracy = accuracy_score(true_choices, pred_choices)
choice_macro_f1 = f1_score(true_choices, pred_choices, average="macro", zero_division=0)

choice_nll = 0.0
for actual, probabilities in zip(true_choices, choice_probabilities):
    probability = float(probabilities[actual].item())
    probability = max(probability, 1e-12)
    choice_nll -= math.log(probability)
choice_nll /= len(true_choices)

choice_cm = confusion_matrix(true_choices, pred_choices, labels=[0, 1, 2, 3])
pred_choice_dist = Counter(pred_choices)

# Baselines
majority_choice_class = max(set(true_choices), key=true_choices.count)
majority_choice_acc = sum(1 for x in true_choices if x == majority_choice_class) / len(true_choices)
num_options = 4
uniform_choice_nll = -math.log(1.0 / num_options)


# ============================================================
# SCORE METRICS
# ============================================================
score_mae = sum(abs(pred - actual) for pred, actual in zip(pred_scores_expected, true_scores)) / len(true_scores)
score_accuracy = accuracy_score(true_scores, pred_scores_rounded)

score_cm = confusion_matrix(true_scores, pred_scores_rounded, labels=[1, 2, 3, 4, 5])
pred_score_dist = Counter(pred_scores_rounded)

# Baselines
always_3_mae = sum(abs(3 - actual) for actual in true_scores) / len(true_scores)
num_scores = 5
uniform_score_nll = -math.log(1.0 / num_scores)


# ============================================================
# PRINT RESULTS
# ============================================================
print("=" * 70)
print("              DRISTI v0.4.1 TEST EVALUATION")
print("=" * 70)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Dataset: {TEST_FILE}")
print(f"Number of examples: {len(test_data)}")
print(f"Checkpoint: {CHECKPOINT}")
print("=" * 70)


print("\n" + "=" * 70)
print("BINARY")
print("=" * 70)
print(f"Accuracy                 : {binary_accuracy:.4f}")
print(f"Brier Score              : {binary_brier:.4f}")
print(f"YES Precision            : {yes_precision:.4f}")
print(f"YES Recall               : {yes_recall:.4f}")
print(f"NO Precision             : {no_precision:.4f}")
print(f"NO Recall                : {no_recall:.4f}")
print("Confusion Matrix (NO=0, YES=1):")
print(binary_cm)
print("-" * 30)
print(f"Baseline (Majority Acc)  : {majority_binary_acc:.4f}")
print(f"Baseline (Const-0.5 Brier): {constant_05_brier:.4f}")


print("\n" + "=" * 70)
print("DYNAMIC CHOICE")
print("=" * 70)
print(f"Accuracy                 : {choice_accuracy:.4f}")
print(f"Macro F1                 : {choice_macro_f1:.4f}")
print(f"NLL                      : {choice_nll:.4f}")
print("Predicted Option Dist    :", dict(sorted(pred_choice_dist.items())))
print("Confusion Matrix:")
print(choice_cm)
print("-" * 30)
print(f"Baseline (Majority Acc)  : {majority_choice_acc:.4f}")
print(f"Baseline (Uniform NLL)   : {uniform_choice_nll:.4f}")


print("\n" + "=" * 70)
print("ORDINAL SCORE")
print("=" * 70)
print(f"Score Accuracy           : {score_accuracy:.4f}")
print(f"Expected MAE             : {score_mae:.4f}")
print("Predicted Score Dist     :", dict(sorted(pred_score_dist.items())))
print("Confusion Matrix (Labels 1 to 5):")
print(score_cm)
print("-" * 30)
print(f"Baseline (Always-3 MAE)  : {always_3_mae:.4f}")
print(f"Baseline (Uniform NLL)   : {uniform_score_nll:.4f}")


print("\n" + "=" * 70)
print("DRISTI v0.4.1 TEST COMPLETE")
print("=" * 70)
