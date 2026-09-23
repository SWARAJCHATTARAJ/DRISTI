import json

import torch
from sklearn.metrics import confusion_matrix

from transformers import AutoTokenizer

from model.dristi_model import DristiModel


MODEL_NAME = "distilbert/distilbert-base-uncased"

TEST_FILE = "data/test.json"

CHECKPOINT_FILE = (
    "checkpoints/dristi_v02_best.pt"
)

MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# LOAD DATA
# =========================================================

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:
    data = json.load(file)


# =========================================================
# LOAD TOKENIZER + MODEL
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = DristiModel(
    num_scores=5
)

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# =========================================================
# STORAGE
# =========================================================

binary_true = []
binary_pred = []

choice_true = []
choice_pred = []

score_true = []
score_pred = []


# =========================================================
# RUN TEST DATA
# =========================================================

with torch.no_grad():

    for item in data:

        question = item["question"]

        options = item["options"]

        # ---------------------------------------------
        # Question
        # ---------------------------------------------

        question_encoding = tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        question_input_ids = (
            question_encoding["input_ids"]
            .to(DEVICE)
        )

        question_attention_mask = (
            question_encoding["attention_mask"]
            .to(DEVICE)
        )

        # ---------------------------------------------
        # Options
        # ---------------------------------------------

        option_texts = []

        for option in options:

            option_texts.append(
                question
                + " [SEP] "
                + option
            )

        option_encoding = tokenizer(
            option_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        option_input_ids = (
            option_encoding["input_ids"]
            .to(DEVICE)
        )

        option_attention_mask = (
            option_encoding["attention_mask"]
            .to(DEVICE)
        )

        # ---------------------------------------------
        # Model
        # ---------------------------------------------

        question_output = (
            model.forward_question(
                question_input_ids,
                question_attention_mask
            )
        )

        option_logits = (
            model.score_options(
                option_input_ids,
                option_attention_mask
            )
        )

        # ---------------------------------------------
        # Binary
        # ---------------------------------------------

        binary_probability = torch.sigmoid(
            question_output["binary"]
        ).item()

        binary_prediction = (
            1
            if binary_probability >= 0.5
            else 0
        )

        binary_true.append(item["yes"])
        binary_pred.append(binary_prediction)

        # ---------------------------------------------
        # Choice
        # ---------------------------------------------

        choice_prediction = int(
            torch.argmax(
                option_logits
            ).item()
        )

        choice_true.append(
            item["choice"]
        )

        choice_pred.append(
            choice_prediction
        )

        # ---------------------------------------------
        # Score
        # ---------------------------------------------

        score_probability = torch.softmax(
            question_output["score"],
            dim=-1
        )[0]

        score_prediction = (
            int(
                torch.argmax(
                    score_probability
                ).item()
            ) + 1
        )

        score_true.append(
            item["score"]
        )

        score_pred.append(
            score_prediction
        )


# =========================================================
# CONFUSION MATRICES
# =========================================================

binary_cm = confusion_matrix(
    binary_true,
    binary_pred,
    labels=[0, 1]
)

choice_cm = confusion_matrix(
    choice_true,
    choice_pred,
    labels=[0, 1, 2, 3]
)

score_cm = confusion_matrix(
    score_true,
    score_pred,
    labels=[1, 2, 3, 4, 5]
)


# =========================================================
# PRINT
# =========================================================

print("=" * 60)
print("            DRISTI v0.2 ERROR ANALYSIS")
print("=" * 60)

print()

print("BINARY CONFUSION MATRIX")
print(
    "Rows = actual, columns = predicted"
)

print(binary_cm)

print()

print("CHOICE CONFUSION MATRIX")
print(
    "Rows = actual, columns = predicted"
)

print(choice_cm)

print()

print("SCORE CONFUSION MATRIX")
print(
    "Rows = actual, columns = predicted"
)

print(score_cm)

print()

print("=" * 60)
print("SCORE ERRORS")
print("=" * 60)

for actual, predicted in zip(
    score_true,
    score_pred
):

    if actual != predicted:

        print(
            f"Actual score={actual}, "
            f"Predicted score={predicted}"
        )

print()
print("=" * 60)