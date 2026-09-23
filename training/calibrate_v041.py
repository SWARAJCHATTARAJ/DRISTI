import json
import torch
from transformers import AutoTokenizer
from torch import nn, optim
from sklearn.metrics import brier_score_loss

from model.dristi_model_v03 import DristiModelV03

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"
CALIBRATION_FILE = "data/calibration_v041.json"
CHECKPOINT = "checkpoints/dristi_v041_best.pt"
OUTPUT_CALIBRATION = "checkpoints/dristi_v041_calibration.json"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODEL & DATA
# ============================================================

with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
    calib_data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = DristiModelV03(model_name=MODEL_NAME, num_scores=5)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ============================================================
# COLLECT LOGITS
# ============================================================

binary_logits_list = []
binary_labels_list = []

choice_logits_list = []
choice_labels_list = []

with torch.inference_mode():
    for item in calib_data:
        question = item["question"]
        options = item["options"]
        actual_binary = int(item["yes"])
        actual_choice = int(item["choice"])

        # Tokenization
        question_tokens = tokenizer(
            question, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        option_texts = [question + " [SEP] " + opt for opt in options]
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

        binary_logits_list.append(question_output["binary"])
        binary_labels_list.append(actual_binary)

        choice_logits_list.append(option_logits)
        choice_labels_list.append(actual_choice)

# Stack logits and labels
binary_logits = torch.cat(binary_logits_list, dim=0).to(device)
binary_labels = torch.tensor(binary_labels_list).to(device)

choice_logits = torch.cat(choice_logits_list, dim=0).to(device)
choice_labels = torch.tensor(choice_labels_list).to(device)

# ============================================================
# TEMPERATURE SCALING
# ============================================================

def calibrate_temperature(logits, labels, criterion):
    """Optimizes the temperature using LBFGS."""
    temperature = nn.Parameter(torch.ones(1).to(device))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=1000)

    def eval():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return temperature.item()

# 1. Binary Calibration
binary_criterion = nn.CrossEntropyLoss()
best_t_binary = calibrate_temperature(binary_logits, binary_labels, binary_criterion)

# 2. Choice Calibration
choice_criterion = nn.CrossEntropyLoss()
best_t_choice = calibrate_temperature(choice_logits, choice_labels, choice_criterion)

# Evaluate Brier Score Before and After for Binary
prob_yes_before = torch.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()
brier_before = brier_score_loss(binary_labels.cpu().numpy(), prob_yes_before)

prob_yes_after = torch.softmax(binary_logits / best_t_binary, dim=1)[:, 1].cpu().numpy()
brier_after = brier_score_loss(binary_labels.cpu().numpy(), prob_yes_after)

# Evaluate NLL Before and After for Choice
nll_before = choice_criterion(choice_logits, choice_labels).item()
nll_after = choice_criterion(choice_logits / best_t_choice, choice_labels).item()

# ============================================================
# SAVE CALIBRATION PARAMETERS
# ============================================================

calibration_params = {
    "binary_temperature": best_t_binary,
    "choice_temperature": best_t_choice,
    "note": "Temperatures are applied as: logits / temperature before softmax."
}

with open(OUTPUT_CALIBRATION, "w") as f:
    json.dump(calibration_params, f, indent=4)

# ============================================================
# PRINT RESULTS
# ============================================================
print("=" * 70)
print("              DRISTI v0.4.1 CALIBRATION")
print("=" * 70)
print(f"Calibration Dataset: {CALIBRATION_FILE}")
print(f"Examples Used: {len(calib_data)}")
print("-" * 70)
print("BINARY LOGITS SCALING:")
print(f"Optimal Temperature (T_binary) : {best_t_binary:.4f}")
print(f"Brier Score Before             : {brier_before:.4f}")
print(f"Brier Score After              : {brier_after:.4f}")
print("-" * 70)
print("CHOICE LOGITS SCALING:")
print(f"Optimal Temperature (T_choice) : {best_t_choice:.4f}")
print(f"NLL Before                     : {nll_before:.4f}")
print(f"NLL After                      : {nll_after:.4f}")
print("=" * 70)
print(f"Calibration parameters successfully saved to:")
print(f"{OUTPUT_CALIBRATION}")
print("=" * 70)
