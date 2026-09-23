import json

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer

from model.dristi_model import DristiModel


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

TEST_FILE = "data/validation.json"

MODEL_FILE = "checkpoints/dristi_model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_LENGTH = 128


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
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# LOAD MODEL
# =========================================================

model = DristiModel(
    num_scores=5
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


# =========================================================
# STORAGE FOR METRICS
# =========================================================

binary_true = []
binary_pred = []
binary_prob = []

choice_true = []
choice_pred = []
choice_probabilities = []

score_true = []
score_pred = []
score_probabilities = []

total_binary_loss = 0.0
total_choice_loss = 0.0
total_score_loss = 0.0


# =========================================================
# EVALUATION
# =========================================================

print("=" * 60)
print("                 DRISTI EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("Examples:", len(data))


with torch.no_grad():

    for item in data:

        question = item["question"]

        options = item["options"]

        # -------------------------------------------------
        # Encode question
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
        # Encode options
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
        # Model prediction
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Binary
        # -------------------------------------------------

        binary_logit = (
            question_output["binary"]
        )

        binary_probability = torch.sigmoid(
            binary_logit
        ).item()

        binary_prediction = (
            1
            if binary_probability >= 0.5
            else 0
        )

        binary_label = item["yes"]

        binary_true.append(
            binary_label
        )

        binary_pred.append(
            binary_prediction
        )

        binary_prob.append(
            binary_probability
        )

        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit,
            torch.tensor(
                [float(binary_label)],
                device=DEVICE
            )
        )

        total_binary_loss += (
            binary_loss.item()
        )

        # -------------------------------------------------
        # Dynamic choice
        # -------------------------------------------------

        probabilities = torch.softmax(
            option_logits,
            dim=0
        )

        predicted_choice = int(
            torch.argmax(probabilities)
        )

        true_choice = item["choice"]

        choice_true.append(
            true_choice
        )

        choice_pred.append(
            predicted_choice
        )

        choice_probabilities.append(
            probabilities.cpu().tolist()
        )

        choice_loss = F.cross_entropy(
            option_logits.unsqueeze(0),
            torch.tensor(
                [true_choice],
                device=DEVICE
            )
        )

        total_choice_loss += (
            choice_loss.item()
        )

        # -------------------------------------------------
        # Score
        # -------------------------------------------------

        score_logits = (
            question_output["score"]
        )

        score_prob = torch.softmax(
            score_logits,
            dim=-1
        )[0]

        predicted_score_index = int(
            torch.argmax(score_prob)
        )

        predicted_score = (
            predicted_score_index + 1
        )

        true_score = item["score"]

        score_true.append(
            true_score
        )

        score_pred.append(
            predicted_score
        )

        score_probabilities.append(
            score_prob.cpu().tolist()
        )

        score_loss = F.cross_entropy(
            score_logits,
            torch.tensor(
                [true_score - 1],
                device=DEVICE
            )
        )

        total_score_loss += (
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

choice_macro_f1 = f1_score(
    choice_true,
    choice_pred,
    average="macro",
    zero_division=0
)

score_mae = sum(
    abs(true - pred)
    for true, pred in zip(
        score_true,
        score_pred
    )
) / len(score_true)


binary_brier = sum(
    (prob - true) ** 2
    for prob, true in zip(
        binary_prob,
        binary_true
    )
) / len(binary_true)


binary_nll = (
    total_binary_loss
    / len(data)
)

choice_nll = (
    total_choice_loss
    / len(data)
)

score_nll = (
    total_score_loss
    / len(data)
)


# =========================================================
# EXPECTED CALIBRATION ERROR
# =========================================================

def calculate_binary_ece(
    probabilities,
    labels,
    bins=10
):

    total = len(probabilities)

    ece = 0.0

    for bin_index in range(bins):

        lower = (
            bin_index / bins
        )

        upper = (
            (bin_index + 1) / bins
        )

        indices = []

        for i, probability in enumerate(
            probabilities
        ):

            if bin_index == bins - 1:

                condition = (
                    lower
                    <= probability
                    <= upper
                )

            else:

                condition = (
                    lower
                    <= probability
                    < upper
                )

            if condition:
                indices.append(i)

        if not indices:
            continue

        average_probability = (
            sum(
                probabilities[i]
                for i in indices
            )
            / len(indices)
        )

        average_accuracy = (
            sum(
                labels[i]
                for i in indices
            )
            / len(indices)
        )

        ece += (
            len(indices)
            / total
        ) * abs(
            average_probability
            - average_accuracy
        )

    return ece


binary_ece = calculate_binary_ece(
    binary_prob,
    binary_true
)


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("BINARY DECISION")
print("=" * 60)

print(
    f"Accuracy      : {binary_accuracy:.4f}"
)

print(
    f"Brier Score   : {binary_brier:.4f}"
)

print(
    f"NLL           : {binary_nll:.4f}"
)

print(
    f"ECE           : {binary_ece:.4f}"
)


print()
print("=" * 60)
print("DYNAMIC CHOICE")
print("=" * 60)

print(
    f"Accuracy      : {choice_accuracy:.4f}"
)

print(
    f"Macro F1      : {choice_macro_f1:.4f}"
)

print(
    f"NLL           : {choice_nll:.4f}"
)


print()
print("=" * 60)
print("SCORE")
print("=" * 60)

print(
    f"MAE           : {score_mae:.4f}"
)

print(
    f"NLL           : {score_nll:.4f}"
)


print()
print("=" * 60)
print("DRISTI EVALUATION COMPLETE")
print("=" * 60)