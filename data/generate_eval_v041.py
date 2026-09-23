import json
import random
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

SEED = 123

rng = random.Random(SEED)

OUTPUT_DIR = Path("data")

VALIDATION_FILE = OUTPUT_DIR / "validation_v041.json"
CALIBRATION_FILE = OUTPUT_DIR / "calibration_v041.json"
TEST_FILE = OUTPUT_DIR / "test_v041.json"


OPTIONS = [
    "Yes",
    "No",
    "Maybe",
    "Unknown",
]


# ============================================================
# QUESTION FORMS
# ============================================================

QUESTION_FORMS = [
    "Is it true that {claim}?",
    "Can we say that {claim}?",
    "Fact check: {claim}.",
    "Is the following statement accurate: {claim}?",
]


# ============================================================
# EVALUATION CLAIMS
# ============================================================
#
# Each score contains:
#
#   10 TRUE claims
#   10 FALSE claims
#
# Total:
#
#   20 claims × 5 scores = 100 claims
#
# Score meaning:
#
# 5 = very clear
# 4 = clear
# 3 = moderate
# 2 = context-dependent
# 1 = highly uncertain / strongly context-dependent
#
# ============================================================


CLAIMS = [

    # ========================================================
    # SCORE 5 — VERY CLEAR
    # ========================================================

    ("Python is a programming language", 1, 5),
    ("C++ is a programming language", 1, 5),
    ("Java is a programming language", 1, 5),
    ("HTML is a markup language", 1, 5),
    ("CSS is used to style web pages", 1, 5),
    ("PostgreSQL is a relational database system", 1, 5),
    ("Excel is spreadsheet software", 1, 5),
    ("Chrome is a web browser", 1, 5),
    ("Windows is an operating system", 1, 5),
    ("a CPU executes program instructions", 1, 5),

    ("Python is a graphics card", 0, 5),
    ("C++ is an image file format", 0, 5),
    ("Java is a database engine", 0, 5),
    ("HTML is a GPU", 0, 5),
    ("CSS is a relational database", 0, 5),
    ("PostgreSQL is a web browser", 0, 5),
    ("Excel is a neural network architecture", 0, 5),
    ("Chrome is a database engine", 0, 5),
    ("Windows is a relational database", 0, 5),
    ("a keyboard is a database management system", 0, 5),


    # ========================================================
    # SCORE 4 — CLEAR
    # ========================================================

    ("JSON is commonly used for structured data exchange", 1, 4),
    ("SQL can be used to query relational databases", 1, 4),
    ("Git is a version control system", 1, 4),
    ("Docker can package applications into containers", 1, 4),
    ("Pandas is commonly used for data manipulation", 1, 4),
    ("NumPy is used for numerical computing", 1, 4),
    ("CUDA can be used for GPU accelerated computing", 1, 4),
    ("DNS can translate domain names into IP addresses", 1, 4),
    ("TCP provides reliable ordered delivery of data", 1, 4),
    ("a firewall can filter network traffic", 1, 4),

    ("JSON is a CPU instruction set", 0, 4),
    ("SQL is primarily an image editing application", 0, 4),
    ("Git is a spreadsheet program", 0, 4),
    ("Docker is a relational database", 0, 4),
    ("Pandas is a GPU hardware device", 0, 4),
    ("NumPy is a web browser", 0, 4),
    ("CUDA is a presentation application", 0, 4),
    ("DNS is a neural network architecture", 0, 4),
    ("TCP is an image file format", 0, 4),
    ("a firewall is a spreadsheet application", 0, 4),


    # ========================================================
    # SCORE 3 — MODERATE
    # ========================================================

    ("a GPU can be useful for parallel numerical workloads", 1, 3),
    ("a database can store user records", 1, 3),
    ("an API can allow software components to communicate", 1, 3),
    ("a web server can respond to HTTP requests", 1, 3),
    ("a compiler can translate source code", 1, 3),
    ("a debugger can help locate software errors", 1, 3),
    ("a VPN can create an encrypted network tunnel", 1, 3),
    ("classification can assign data to categories", 1, 3),
    ("regression can be used to predict numerical values", 1, 3),
    ("clustering can group similar data points", 1, 3),

    ("a GPU is always faster than a CPU for every workload", 0, 3),
    ("a database automatically makes every stored value correct", 0, 3),
    ("an API is always a network protocol", 0, 3),
    ("a web server can only serve HTML files", 0, 3),
    ("a compiler can always understand incorrect source code", 0, 3),
    ("a debugger automatically fixes every software bug", 0, 3),
    ("a VPN makes every application completely anonymous", 0, 3),
    ("classification always requires exactly two categories", 0, 3),
    ("regression is only useful for text labels", 0, 3),
    ("clustering always requires manually assigned class labels", 0, 3),


    # ========================================================
    # SCORE 2 — CONTEXT DEPENDENT
    # ========================================================

    ("Python can be a useful choice for many data analysis tasks", 1, 2),
    ("a GPU may improve some computational workloads", 1, 2),
    ("a database may be appropriate for structured records", 1, 2),
    ("Docker can simplify application deployment in many environments", 1, 2),
    ("a VPN can improve privacy in some network situations", 1, 2),
    ("caching can improve performance in some applications", 1, 2),
    ("parallel processing can reduce runtime for suitable workloads", 1, 2),
    ("cloud computing can be useful for scalable workloads", 1, 2),
    ("a compiler can be helpful in some software development workflows", 1, 2),
    ("automation can reduce repetitive manual work", 1, 2),

    ("Python is always the best language for every data task", 0, 2),
    ("a GPU always improves application performance", 0, 2),
    ("a database is always the best way to store every kind of data", 0, 2),
    ("Docker always makes every application faster", 0, 2),
    ("a VPN always guarantees complete privacy", 0, 2),
    ("caching always improves performance", 0, 2),
    ("parallel processing always reduces runtime", 0, 2),
    ("cloud computing is always cheaper than local computing", 0, 2),
    ("a compiler automatically improves program design", 0, 2),
    ("automation completely removes the need for human review", 0, 2),


    # ========================================================
    # SCORE 1 — HIGHLY CONTEXTUAL
    # ========================================================

    ("whether Python is suitable depends on the task requirements", 1, 1),
    ("whether a GPU is useful depends on the workload", 1, 1),
    ("whether a database is appropriate depends on the data requirements", 1, 1),
    ("whether Docker is useful depends on the deployment environment", 1, 1),
    ("whether a VPN is beneficial depends on the network situation", 1, 1),
    ("whether caching helps depends on the application workload", 1, 1),
    ("whether parallel processing helps depends on the computation", 1, 1),
    ("whether cloud computing is suitable depends on the workload and constraints", 1, 1),
    ("whether automation is appropriate depends on the process", 1, 1),
    ("whether a compiler is helpful depends on the programming workflow", 1, 1),

    ("this software will definitely solve every computing problem", 0, 1),
    ("this computer configuration is guaranteed to be optimal", 0, 1),
    ("this algorithm will always produce the best possible result", 0, 1),
    ("this network design is guaranteed to be secure", 0, 1),
    ("this machine learning model will always be correct", 0, 1),
    ("this database design is perfect for every application", 0, 1),
    ("this program will never contain a bug", 0, 1),
    ("this cloud architecture is guaranteed to minimize all costs", 0, 1),
    ("this optimization method will always find the global optimum", 0, 1),
    ("this security configuration can never be bypassed", 0, 1),
]


# ============================================================
# VERIFY CLAIMS
# ============================================================

assert len(CLAIMS) == 100


score_counts = Counter(
    score
    for _, _, score in CLAIMS
)

yes_counts = Counter(
    yes
    for _, yes, _ in CLAIMS
)


for score in range(1, 6):

    assert score_counts[score] == 20


assert yes_counts[0] == 50
assert yes_counts[1] == 50


# ============================================================
# BUILD QUESTIONS
# ============================================================

examples = []

for index, (
    claim,
    yes_label,
    score
) in enumerate(CLAIMS):

    form = QUESTION_FORMS[
        index % len(QUESTION_FORMS)
    ]

    question = form.format(
        claim=claim
    )

    examples.append(
        {
            "question": question,
            "yes": yes_label,
            "score": score,
        }
    )


# ============================================================
# BALANCED CORRECT OPTION POSITIONS
# ============================================================

# Across all 100 examples:
#
# 25 correct answers at each position.

position_schedule = []

for position in range(4):

    position_schedule.extend(
        [position] * 25
    )

rng.shuffle(
    position_schedule
)


# ============================================================
# BUILD OPTIONS
# ============================================================

dataset = []

for example, correct_position in zip(
    examples,
    position_schedule
):

    correct_answer = (
        "Yes"
        if example["yes"] == 1
        else "No"
    )

    other_options = [
        option
        for option in OPTIONS
        if option != correct_answer
    ]

    option_list = [None] * 4

    option_list[
        correct_position
    ] = correct_answer

    other_index = 0

    for position in range(4):

        if option_list[position] is None:

            option_list[position] = (
                other_options[
                    other_index
                ]
            )

            other_index += 1

    choice = option_list.index(
        correct_answer
    )

    dataset.append(
        {
            "question": example["question"],
            "options": option_list,
            "choice": choice,
            "yes": example["yes"],
            "score": example["score"],
        }
    )


# ============================================================
# STRATIFIED SPLIT
# ============================================================
#
# For EACH score:
#
# Validation:
#   3 YES + 3 NO = 6
#
# Calibration:
#   3 YES + 3 NO = 6
#
# Test:
#   4 YES + 4 NO = 8
#
# Across 5 scores:
#
# Validation = 30
# Calibration = 30
# Test = 40
#
# Each split:
#
# YES = 50%
# NO  = 50%
#
# Each score = balanced.
#
# ============================================================

groups = {}

for score in range(1, 6):

    score_items = [
        item
        for item in dataset
        if item["score"] == score
    ]

    yes_items = [
        item
        for item in score_items
        if item["yes"] == 1
    ]

    no_items = [
        item
        for item in score_items
        if item["yes"] == 0
    ]

    assert len(yes_items) == 10
    assert len(no_items) == 10

    rng.shuffle(
        yes_items
    )

    rng.shuffle(
        no_items
    )

    validation_group = (
        yes_items[:3]
        + no_items[:3]
    )

    calibration_group = (
        yes_items[3:6]
        + no_items[3:6]
    )

    test_group = (
        yes_items[6:10]
        + no_items[6:10]
    )

    groups[score] = {
        "validation": validation_group,
        "calibration": calibration_group,
        "test": test_group,
    }


validation_data = []
calibration_data = []
test_data = []


for score in range(1, 6):

    validation_data.extend(
        groups[score]["validation"]
    )

    calibration_data.extend(
        groups[score]["calibration"]
    )

    test_data.extend(
        groups[score]["test"]
    )


rng.shuffle(
    validation_data
)

rng.shuffle(
    calibration_data
)

rng.shuffle(
    test_data
)


# ============================================================
# VERIFY SPLITS
# ============================================================

assert len(validation_data) == 30
assert len(calibration_data) == 30
assert len(test_data) == 40


def verify_split(
    name,
    data,
    expected_size,
    expected_yes,
    expected_no
):

    assert len(data) == expected_size

    yes_count = Counter(
        item["yes"]
        for item in data
    )

    score_count = Counter(
        item["score"]
        for item in data
    )

    assert yes_count[1] == expected_yes
    assert yes_count[0] == expected_no

    if expected_size == 30:

        for score in range(1, 6):

            assert score_count[score] == 6

    elif expected_size == 40:

        for score in range(1, 6):

            assert score_count[score] == 8

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Examples:",
        len(data)
    )

    print(
        "NO:",
        yes_count[0]
    )

    print(
        "YES:",
        yes_count[1]
    )

    print("Scores:")

    for score in range(1, 6):

        print(
            f"  {score}: "
            f"{score_count[score]}"
        )


verify_split(
    "VALIDATION",
    validation_data,
    30,
    15,
    15
)

verify_split(
    "CALIBRATION",
    calibration_data,
    30,
    15,
    15
)

verify_split(
    "TEST",
    test_data,
    40,
    20,
    20
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

all_questions = [
    item["question"]
    for item in dataset
]

assert len(all_questions) == len(
    set(all_questions)
)


# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

validation_questions = {
    item["question"]
    for item in validation_data
}

calibration_questions = {
    item["question"]
    for item in calibration_data
}

test_questions = {
    item["question"]
    for item in test_data
}


assert (
    validation_questions
    & calibration_questions
) == set()

assert (
    validation_questions
    & test_questions
) == set()

assert (
    calibration_questions
    & test_questions
) == set()


# ============================================================
# SAVE FILES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_json(
    path,
    data
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


save_json(
    VALIDATION_FILE,
    validation_data
)

save_json(
    CALIBRATION_FILE,
    calibration_data
)

save_json(
    TEST_FILE,
    test_data
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("DRISTI v0.4.1 EVALUATION DATA GENERATED")
print("=" * 70)

print()
print("Created:")
print(
    VALIDATION_FILE
)

print(
    CALIBRATION_FILE
)

print(
    TEST_FILE
)

print()
print(
    "Original validation.json, calibration.json,"
)

print(
    "and test.json were NOT modified."
)

print("=" * 70)