import json
import math
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import softmax

from model.dristi_model_v03 import DristiModelV03


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

TEST_FILE = "data/test.json"

CHECKPOINT = "checkpoints/dristi_v031_best.pt"

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
    map_location=device
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

        question_input_ids = (
            question_tokens["input_ids"]
            .to(device)
        )

        question_attention_mask = (
            question_tokens["attention_mask"]
            .to(device)
        )


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

        # Shape:
        # [num_options, sequence_length]
        #
        # Add batch dimension:
        # [1, num_options, sequence_length]

        option_input_ids = (
            option_tokens["input_ids"]
            .unsqueeze(0)
            .to(device)
        )

        option_attention_mask = (
            option_tokens["attention_mask"]
            .unsqueeze(0)
            .to(device)
        )


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

        binary_probs = softmax(
            binary_logits,
            dim=1
        )

        probability_yes = float(
            binary_probs[0, 1].item()
        )

        binary_prediction = int(
            torch.argmax(
                binary_logits,
                dim=1
            ).item()
        )


        # ----------------------------------------------------
        # CHOICE PREDICTION
        # ----------------------------------------------------

        option_logits = model.score_options(
            option_input_ids,
            option_attention_mask
        )

        choice_probs = softmax(
            option_logits,
            dim=1
        )

        choice_prediction = int(
            torch.argmax(
                choice_probs,
                dim=1
            ).item()
        )


        # ----------------------------------------------------
        # ORDINAL SCORE
        # ----------------------------------------------------
        #
        # Model predicts:
        #
        # P(score >= 2)
        # P(score >= 3)
        # P(score >= 4)
        # P(score >= 5)
        #
        # E[S] =
        # 1 + P(score>=2)
        #   + P(score>=3)
        #   + P(score>=4)
        #   + P(score>=5)
        #

        threshold_probabilities = torch.sigmoid(
            ordinal_logits
        )

        expected_score = float(
            1.0
            + threshold_probabilities.sum().item()
        )

        rounded_score = int(
            round(expected_score)
        )

        rounded_score = max(
            1,
            min(5, rounded_score)
        )


        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        true_binary.append(
            actual_binary
        )

        pred_binary.append(
            binary_prediction
        )

        binary_probabilities.append(
            probability_yes
        )


        true_choices.append(
            actual_choice
        )

        pred_choices.append(
            choice_prediction
        )

        choice_probabilities.append(
            choice_probs[0].cpu()
        )


        true_scores.append(
            actual_score
        )

        pred_scores_expected.append(
            expected_score
        )

        pred_scores_rounded.append(
            rounded_score
        )


# ============================================================
# BINARY METRICS
# ============================================================

binary_accuracy = accuracy_score(
    true_binary,
    pred_binary
)

binary_brier = sum(
    (prob - actual) ** 2
    for prob, actual in zip(
        binary_probabilities,
        true_binary
    )
) / len(true_binary)


# ============================================================
# CHOICE METRICS
# ============================================================

choice_accuracy = accuracy_score(
    true_choices,
    pred_choices
)

choice_macro_f1 = f1_score(
    true_choices,
    pred_choices,
    average="macro"
)


# Choice NLL
choice_nll = 0.0

for actual, probabilities in zip(
    true_choices,
    choice_probabilities
):

    probability = float(
        probabilities[actual].item()
    )

    probability = max(
        probability,
        1e-12
    )

    choice_nll -= math.log(
        probability
    )

choice_nll /= len(true_choices)


# ============================================================
# SCORE METRICS
# ============================================================

score_mae = sum(
    abs(pred - actual)
    for pred, actual in zip(
        pred_scores_expected,
        true_scores
    )
) / len(true_scores)


score_accuracy = accuracy_score(
    true_scores,
    pred_scores_rounded
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("              DRISTI v0.3.1 TEST EVALUATION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(f"Test examples: {len(test_data)}")

print("=" * 70)


print("\n")
print("=" * 70)
print("BINARY")
print("=" * 70)

print(
    f"Accuracy      : {binary_accuracy:.4f}"
)

print(
    f"Brier Score   : {binary_brier:.4f}"
)


print("\n")
print("=" * 70)
print("DYNAMIC CHOICE")
print("=" * 70)

print(
    f"Accuracy      : {choice_accuracy:.4f}"
)

print(
    f"Macro F1      : {choice_macro_f1:.4f}"
)

print(
    f"NLL           : {choice_nll:.4f}"
)


print("\n")
print("=" * 70)
print("ORDINAL SCORE")
print("=" * 70)

print(
    f"Expected MAE  : {score_mae:.4f}"
)

print(
    f"Score Accuracy: {score_accuracy:.4f}"
)


print("\n")
print("=" * 70)
print("DRISTI v0.3.1 TEST COMPLETE")
print("=" * 70)