import json
import math
from collections import Counter


TEST_FILE = "data/test.json"


with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


# =========================================================
# BINARY BASELINE
# =========================================================

binary_labels = [
    item["yes"]
    for item in data
]

majority_binary = (
    1
    if sum(binary_labels) >= len(binary_labels) / 2
    else 0
)

binary_accuracy = sum(
    label == majority_binary
    for label in binary_labels
) / len(binary_labels)


# Constant 0.5 probability
binary_brier = sum(
    (0.5 - label) ** 2
    for label in binary_labels
) / len(binary_labels)


# =========================================================
# CHOICE BASELINE
# =========================================================

choice_labels = [
    item["choice"]
    for item in data
]

choice_counts = Counter(
    choice_labels
)

majority_choice = (
    choice_counts.most_common(1)[0][0]
)


choice_accuracy = sum(
    label == majority_choice
    for label in choice_labels
) / len(choice_labels)


# Uniform NLL for 4 choices
choice_nll = math.log(4)


# =========================================================
# SCORE BASELINE
# =========================================================

score_labels = [
    item["score"]
    for item in data
]

median_score = 3


score_mae = sum(
    abs(
        label - median_score
    )
    for label in score_labels
) / len(score_labels)


# Uniform probability over scores 1-5
score_nll = math.log(5)


# =========================================================
# PRINT
# =========================================================

print("=" * 60)
print("             DRISTI BASELINE RESULTS")
print("=" * 60)

print()
print("BINARY")
print("Majority accuracy:", binary_accuracy)
print("Constant-0.5 Brier:", binary_brier)

print()
print("CHOICE")
print("Majority accuracy:", choice_accuracy)
print("Uniform-choice NLL:", choice_nll)

print()
print("SCORE")
print("Always predict 3 MAE:", score_mae)
print("Uniform-score NLL:", score_nll)

print()
print("=" * 60)