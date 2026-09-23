import json

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error
)

from transformers import AutoTokenizer

from model.dristi_model_v03 import DristiModelV03


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

TEST_FILE = "data/test.json"

CHECKPOINT_FILE = (
    "checkpoints/dristi_v03_best.pt"
)

MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD TEST DATA
# =========================================================

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# LOAD MODEL
# =========================================================

model = DristiModelV03(
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
binary_probability = []

choice_true = []
choice_pred = []

score_true = []
score_pred = []
score_expected = []


# =========================================================
# EVALUATION
# =========================================================

print("=" * 65)
print("              DRISTI v0.3 TEST EVALUATION")
print("=" * 65)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Test examples:",
    len(data)
)

print("=" * 65)


with torch.no_grad():

    for item in data:

        question = item["question"]

        options = item["options"]


        # =================================================
        # QUESTION
        # =================================================

        question_encoding = tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        question_input_ids = (
            question_encoding[
                "input_ids"
            ].to(DEVICE)
        )

        question_attention_mask = (
            question_encoding[
                "attention_mask"
            ].to(DEVICE)
        )


        # =================================================
        # OPTIONS
        # =================================================

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
            option_encoding[
                "input_ids"
            ]
            .unsqueeze(0)
            .to(DEVICE)
        )

        option_attention_mask = (
            option_encoding[
                "attention_mask"
            ]
            .unsqueeze(0)
            .to(DEVICE)
        )


        # =================================================
        # MODEL
        # =================================================

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


        # =================================================
        # BINARY
        # =================================================

        binary_prob = torch.softmax(
            question_output["binary"],
            dim=1
        )[0, 1].item()


        binary_prediction = (
            1
            if binary_prob >= 0.5
            else 0
        )


        binary_true.append(
            item["yes"]
        )

        binary_pred.append(
            binary_prediction
        )

        binary_probability.append(
            binary_prob
        )


        # =================================================
        # CHOICE
        # =================================================

        choice_prob = torch.softmax(
            option_logits[0],
            dim=0
        )


        choice_prediction = int(
            torch.argmax(
                choice_prob
            ).item()
        )


        choice_true.append(
            item["choice"]
        )

        choice_pred.append(
            choice_prediction
        )


        # =================================================
        # ORDINAL SCORE
        # =================================================

        ordinal_logits = (
            question_output["ordinal"]
        )


        threshold_probabilities = torch.sigmoid(
            ordinal_logits
        )[0]


        # E[S]
        expected_score = (
            1.0
            + threshold_probabilities.sum().item()
        )


        predicted_score = int(
            max(
                1,
                min(
                    5,
                    round(expected_score)
                )
            )
        )


        score_true.append(
            item["score"]
        )

        score_pred.append(
            predicted_score
        )

        score_expected.append(
            expected_score
        )


# =========================================================
# METRICS
# =========================================================

binary_accuracy = accuracy_score(
    binary_true,
    binary_pred
)

choice_accuracy = accuracy_score(
    choice_true,
    choice_pred
)

choice_macro_f1 = f1_score(
    choice_true,
    choice_pred,
    average="macro",
    zero_division=0
)

score_mae = mean_absolute_error(
    score_true,
    score_expected
)

score_classification_accuracy = accuracy_score(
    score_true,
    score_pred
)


# =========================================================
# BRIER
# =========================================================

binary_brier = sum(
    (
        probability - label
    ) ** 2

    for probability, label
    in zip(
        binary_probability,
        binary_true
    )
) / len(binary_true)


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("=" * 65)
print("BINARY")
print("=" * 65)

print(
    f"Accuracy      : {binary_accuracy:.4f}"
)

print(
    f"Brier Score   : {binary_brier:.4f}"
)


print()
print("=" * 65)
print("DYNAMIC CHOICE")
print("=" * 65)

print(
    f"Accuracy      : {choice_accuracy:.4f}"
)

print(
    f"Macro F1      : {choice_macro_f1:.4f}"
)


print()
print("=" * 65)
print("ORDINAL SCORE")
print("=" * 65)

print(
    f"Expected MAE  : {score_mae:.4f}"
)

print(
    f"Score Accuracy: "
    f"{score_classification_accuracy:.4f}"
)


print()
print("=" * 65)
print("DRISTI v0.3 TEST COMPLETE")
print("=" * 65)