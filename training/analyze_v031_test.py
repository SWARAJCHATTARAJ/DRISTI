import json
import math

import torch
from transformers import AutoTokenizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
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

print("=" * 70)
print("        DRISTI v0.3.1 TEST ERROR ANALYSIS")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:
    test_data = json.load(file)

print(f"Test examples: {len(test_data)}")


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
# STORAGE
# ============================================================

true_binary = []
pred_binary = []
binary_yes_probs = []

true_choices = []
pred_choices = []
choice_probs_list = []

true_scores = []
pred_scores_expected = []
pred_scores_rounded = []

binary_errors = []
choice_errors = []
score_errors = []


# ============================================================
# EVALUATION
# ============================================================

with torch.inference_mode():

    for index, item in enumerate(test_data):

        question = item["question"]
        options = item["options"]

        actual_binary = int(item["yes"])
        actual_choice = int(item["choice"])
        actual_score = int(item["score"])


        # ----------------------------------------------------
        # Question tokenization
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
        # Option tokenization
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
        # Question prediction
        # ----------------------------------------------------

        question_output = model.forward_question(
            question_input_ids,
            question_attention_mask
        )

        binary_logits = question_output["binary"]
        ordinal_logits = question_output["ordinal"]


        # ----------------------------------------------------
        # Binary
        # ----------------------------------------------------

        binary_probabilities = softmax(
            binary_logits,
            dim=1
        )

        yes_probability = float(
            binary_probabilities[0, 1].item()
        )

        binary_prediction = int(
            torch.argmax(
                binary_logits,
                dim=1
            ).item()
        )


        # ----------------------------------------------------
        # Dynamic choice
        # ----------------------------------------------------

        option_logits = model.score_options(
            option_input_ids,
            option_attention_mask
        )

        choice_probabilities = softmax(
            option_logits,
            dim=1
        )

        choice_prediction = int(
            torch.argmax(
                choice_probabilities,
                dim=1
            ).item()
        )


        # ----------------------------------------------------
        # Ordinal score
        # ----------------------------------------------------

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
        # Store results
        # ----------------------------------------------------

        true_binary.append(actual_binary)
        pred_binary.append(binary_prediction)
        binary_yes_probs.append(yes_probability)

        true_choices.append(actual_choice)
        pred_choices.append(choice_prediction)
        choice_probs_list.append(
            choice_probabilities[0].cpu()
        )

        true_scores.append(actual_score)
        pred_scores_expected.append(expected_score)
        pred_scores_rounded.append(rounded_score)


        # ----------------------------------------------------
        # Store errors
        # ----------------------------------------------------

        if binary_prediction != actual_binary:
            binary_errors.append({
                "index": index,
                "question": question,
                "actual": actual_binary,
                "predicted": binary_prediction,
                "yes_probability": yes_probability
            })

        if choice_prediction != actual_choice:
            choice_errors.append({
                "index": index,
                "question": question,
                "options": options,
                "actual": actual_choice,
                "predicted": choice_prediction,
                "probabilities": (
                    choice_probabilities[0]
                    .cpu()
                    .tolist()
                )
            })

        if rounded_score != actual_score:
            score_errors.append({
                "index": index,
                "question": question,
                "actual": actual_score,
                "predicted": rounded_score,
                "expected": expected_score
            })


# ============================================================
# BINARY ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("BINARY CONFUSION MATRIX")
print("=" * 70)

binary_cm = confusion_matrix(
    true_binary,
    pred_binary,
    labels=[0, 1]
)

print("Rows    = Actual")
print("Columns = Predicted")
print()

print(binary_cm)

print()
print("Labels:")
print("0 = NO")
print("1 = YES")


print("\n")
print("=" * 70)
print("BINARY CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        true_binary,
        pred_binary,
        labels=[0, 1],
        target_names=["NO", "YES"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# BINARY BRIER
# ============================================================

binary_brier = sum(
    (prob - actual) ** 2
    for prob, actual in zip(
        binary_yes_probs,
        true_binary
    )
) / len(true_binary)

print(
    f"Binary Brier Score: {binary_brier:.4f}"
)


# ============================================================
# BINARY DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("BINARY PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    f"Actual NO : {true_binary.count(0)}"
)

print(
    f"Actual YES: {true_binary.count(1)}"
)

print(
    f"Predicted NO : {pred_binary.count(0)}"
)

print(
    f"Predicted YES: {pred_binary.count(1)}"
)


# ============================================================
# CHOICE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("DYNAMIC CHOICE CONFUSION MATRIX")
print("=" * 70)

choice_cm = confusion_matrix(
    true_choices,
    pred_choices,
    labels=[0, 1, 2, 3]
)

print("Rows    = Actual")
print("Columns = Predicted")
print()

print(choice_cm)

print()
print("Choice positions:")
print("0 = Option 1")
print("1 = Option 2")
print("2 = Option 3")
print("3 = Option 4")


# ============================================================
# CHOICE CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CHOICE CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        true_choices,
        pred_choices,
        labels=[0, 1, 2, 3],
        target_names=[
            "Option 1",
            "Option 2",
            "Option 3",
            "Option 4"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CHOICE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("CHOICE DISTRIBUTION")
print("=" * 70)

print("Actual:")

for i in range(4):
    print(
        f"  Option {i + 1}: "
        f"{true_choices.count(i)}"
    )

print()

print("Predicted:")

for i in range(4):
    print(
        f"  Option {i + 1}: "
        f"{pred_choices.count(i)}"
    )


# ============================================================
# CHOICE NLL
# ============================================================

choice_nll = 0.0

for actual, probabilities in zip(
    true_choices,
    choice_probs_list
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

print()
print(
    f"Choice NLL: {choice_nll:.4f}"
)


# ============================================================
# SCORE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("SCORE CONFUSION MATRIX")
print("=" * 70)

score_cm = confusion_matrix(
    true_scores,
    pred_scores_rounded,
    labels=[1, 2, 3, 4, 5]
)

print("Rows    = Actual score")
print("Columns = Predicted score")
print()

print(score_cm)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("SCORE DISTRIBUTION")
print("=" * 70)

print("Actual:")

for score in range(1, 6):
    print(
        f"  Score {score}: "
        f"{true_scores.count(score)}"
    )

print()

print("Predicted:")

for score in range(1, 6):
    print(
        f"  Score {score}: "
        f"{pred_scores_rounded.count(score)}"
    )


# ============================================================
# SCORE MAE
# ============================================================

score_mae = sum(
    abs(pred - actual)
    for pred, actual in zip(
        pred_scores_expected,
        true_scores
    )
) / len(true_scores)

print()
print(
    f"Score Expected MAE: {score_mae:.4f}"
)


# ============================================================
# ERROR COUNTS
# ============================================================

print("\n")
print("=" * 70)
print("ERROR COUNTS")
print("=" * 70)

print(
    f"Binary errors : {len(binary_errors)}"
)

print(
    f"Choice errors : {len(choice_errors)}"
)

print(
    f"Score errors  : {len(score_errors)}"
)


# ============================================================
# BINARY WRONG EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("BINARY WRONG EXAMPLES")
print("=" * 70)

for error in binary_errors[:10]:

    print()
    print(f"Test index: {error['index']}")
    print(f"Question  : {error['question']}")
    print(f"Actual    : {error['actual']}")
    print(f"Predicted : {error['predicted']}")
    print(
        f"P(YES)    : "
        f"{error['yes_probability']:.4f}"
    )


# ============================================================
# CHOICE WRONG EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("CHOICE WRONG EXAMPLES")
print("=" * 70)

for error in choice_errors[:10]:

    print()
    print(f"Test index: {error['index']}")
    print(f"Question  : {error['question']}")

    print("Options:")

    for i, option in enumerate(
        error["options"]
    ):
        print(
            f"  {i + 1}. {option}"
        )

    print(
        f"Actual option    : "
        f"{error['actual'] + 1}"
    )

    print(
        f"Predicted option : "
        f"{error['predicted'] + 1}"
    )

    print("Probabilities:")

    for i, probability in enumerate(
        error["probabilities"]
    ):
        print(
            f"  Option {i + 1}: "
            f"{probability:.4f}"
        )


# ============================================================
# SCORE WRONG EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("SCORE WRONG EXAMPLES")
print("=" * 70)

for error in score_errors[:10]:

    print()
    print(f"Test index: {error['index']}")
    print(f"Question  : {error['question']}")
    print(f"Actual    : {error['actual']}")
    print(f"Predicted : {error['predicted']}")
    print(
        f"Expected  : "
        f"{error['expected']:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("DRISTI v0.3.1 ERROR ANALYSIS COMPLETE")
print("=" * 70)