import json

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error
)

from transformers import AutoTokenizer

from model.dristi_model import DristiModel


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

TEST_FILE = "data/test.json"

CHECKPOINT_FILE = (
    "checkpoints/dristi_v02_best.pt"
)

MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# MODEL
# =========================================================

model = DristiModel(
    num_scores=5
)


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE
)


# v0.2 checkpoint is a dictionary.
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
binary_probs = []

choice_true = []
choice_pred = []
choice_losses = []

score_true = []
score_pred = []
score_losses = []


# =========================================================
# EVALUATION
# =========================================================

print("=" * 60)
print("             DRISTI v0.2 TEST EVALUATION")
print("=" * 60)

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


with torch.no_grad():

    for item in data:

        question = item["question"]

        options = item["options"]


        # -------------------------------------------------
        # QUESTION
        # -------------------------------------------------

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


        # -------------------------------------------------
        # OPTIONS
        # -------------------------------------------------

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


        # -------------------------------------------------
        # QUESTION HEADS
        # -------------------------------------------------

        question_output = (
            model.forward_question(
                question_input_ids,
                question_attention_mask
            )
        )


        # -------------------------------------------------
        # OPTION SCORING
        # -------------------------------------------------

        option_logits = (
            model.score_options(
                option_input_ids,
                option_attention_mask
            )
        )


        # =================================================
        # BINARY
        # =================================================

        binary_probability = torch.sigmoid(
            question_output["binary"]
        ).item()


        binary_prediction = (
            1
            if binary_probability >= 0.5
            else 0
        )


        binary_true.append(
            item["yes"]
        )

        binary_pred.append(
            binary_prediction
        )

        binary_probs.append(
            binary_probability
        )


        # =================================================
        # CHOICE
        # =================================================

        choice_probability = torch.softmax(
            option_logits,
            dim=0
        )


        choice_prediction = int(
            torch.argmax(
                choice_probability
            ).item()
        )


        choice_true.append(
            item["choice"]
        )

        choice_pred.append(
            choice_prediction
        )


        choice_loss = F.cross_entropy(
            option_logits.unsqueeze(0),
            torch.tensor(
                [item["choice"]],
                device=DEVICE
            )
        )


        choice_losses.append(
            choice_loss.item()
        )


        # =================================================
        # SCORE
        # =================================================

        score_probability = torch.softmax(
            question_output["score"],
            dim=-1
        )[0]


        score_prediction = (
            int(torch.argmax(
                score_probability
            ).item())
            + 1
        )


        score_true.append(
            item["score"]
        )

        score_pred.append(
            score_prediction
        )


        score_loss = F.cross_entropy(
            question_output["score"],
            torch.tensor(
                [item["score"] - 1],
                device=DEVICE
            )
        )


        score_losses.append(
            score_loss.item()
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


choice_f1 = f1_score(
    choice_true,
    choice_pred,
    average="macro",
    zero_division=0
)


score_mae = mean_absolute_error(
    score_true,
    score_pred
)


binary_brier = sum(
    (
        probability - label
    ) ** 2

    for probability, label
    in zip(
        binary_probs,
        binary_true
    )
) / len(binary_true)


choice_nll = sum(
    choice_losses
) / len(choice_losses)


score_nll = sum(
    score_losses
) / len(score_losses)


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("BINARY")
print("=" * 60)

print(
    f"Accuracy    : {binary_accuracy:.4f}"
)

print(
    f"Brier Score : {binary_brier:.4f}"
)


print()
print("=" * 60)
print("DYNAMIC CHOICE")
print("=" * 60)

print(
    f"Accuracy    : {choice_accuracy:.4f}"
)

print(
    f"Macro F1    : {choice_f1:.4f}"
)

print(
    f"NLL         : {choice_nll:.4f}"
)


print()
print("=" * 60)
print("SCORE")
print("=" * 60)

print(
    f"MAE         : {score_mae:.4f}"
)

print(
    f"NLL         : {score_nll:.4f}"
)


print()
print("=" * 60)
print("DRISTI v0.2 TEST COMPLETE")
print("=" * 60)