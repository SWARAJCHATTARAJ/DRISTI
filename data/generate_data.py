import json
import random
import shutil
import string

from collections import Counter
from itertools import permutations
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SEED = 42

rng = random.Random(SEED)

OUTPUT_FILE = Path("data/train.json")

BACKUP_FILE = Path("data/train_v041_backup.json")

EVAL_FILES = [
    Path("data/validation.json"),
    Path("data/calibration.json"),
    Path("data/test.json"),
    Path("data/validation_v041.json"),
    Path("data/calibration_v041.json"),
    Path("data/test_v041.json"),
]

VARIANTS_PER_CLAIM = 10

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
    "Consider this statement: {claim}. Is it correct?",
    "Fact check: {claim}.",
    "Is the following statement accurate: {claim}?",
    "Would you mark this statement as true: {claim}?",
    "Can this claim be considered correct: {claim}?",
    "Does this statement hold: {claim}?",
    "Is the claim correct: {claim}?",
    "Should this statement be marked true: {claim}?",
]


# ============================================================
# CLAIM DATA
# ============================================================
#
# Format:
#
# (claim, yes_label, score)
#
# yes_label:
#   1 = TRUE
#   0 = FALSE
#
# score:
#   5 = very clear
#   4 = clear
#   3 = moderately clear
#   2 = context dependent
#   1 = highly context dependent
#
# Exactly:
#
#   24 claims for score 1
#   24 claims for score 2
#   24 claims for score 3
#   24 claims for score 4
#   24 claims for score 5
#
# Total = 120 unique claims
#
# Each score has:
#
#   12 TRUE
#   12 FALSE
#
# ============================================================

CLAIMS = [

    # ========================================================
    # SCORE 5 — VERY CLEAR
    # ========================================================

    ("JavaScript is a programming language", 1, 5),
    ("Rust is a programming language", 1, 5),
    ("Go is a programming language", 1, 5),
    ("Ruby is a programming language", 1, 5),
    ("PHP is a programming language", 1, 5),
    ("SQLite is a lightweight embedded database engine", 1, 5),
    ("MySQL is a relational database management system", 1, 5),
    ("MongoDB is a document-oriented database system", 1, 5),
    ("Redis can be used as a key-value data store", 1, 5),
    ("a keyboard is an input device", 1, 5),
    ("a monitor is an output device", 1, 5),
    ("an SSD is a storage device", 1, 5),

    ("JavaScript is a graphics card", 0, 5),
    ("Rust is a spreadsheet application", 0, 5),
    ("Go is a raster image format", 0, 5),
    ("Ruby is a database table", 0, 5),
    ("PHP is a relational database engine", 0, 5),
    ("SQLite is a graphics editor", 0, 5),
    ("MySQL is a web browser", 0, 5),
    ("MongoDB is a presentation program", 0, 5),
    ("Redis is a monitor device", 0, 5),
    ("a keyboard is a database server", 0, 5),
    ("a monitor is a programming language", 0, 5),
    ("an SSD is a markup language", 0, 5),


    # ========================================================
    # SCORE 4 — CLEAR
    # ========================================================

    ("XML can represent structured information", 1, 4),
    ("IPv4 addresses use 32 bits", 1, 4),
    ("IPv6 addresses use 128 bits", 1, 4),
    ("GitHub can host software repositories", 1, 4),
    ("Kubernetes can orchestrate containerized workloads", 1, 4),
    ("OpenCV can be used for computer vision tasks", 1, 4),
    ("Matplotlib can create data visualizations", 1, 4),
    ("a router can forward packets between networks", 1, 4),
    ("a web server can receive HTTP requests", 1, 4),
    ("a printer is an output device", 1, 4),
    ("a motherboard connects major computer components", 1, 4),
    ("a USB port can connect compatible peripheral devices", 1, 4),

    ("XML is a graphics processing unit", 0, 4),
    ("IPv4 addresses use 128 bits", 0, 4),
    ("IPv6 addresses use 32 bits", 0, 4),
    ("GitHub is a CPU instruction set", 0, 4),
    ("Kubernetes is an image file format", 0, 4),
    ("OpenCV is a database query language", 0, 4),
    ("Matplotlib is a hard disk", 0, 4),
    ("a router is a spreadsheet application", 0, 4),
    ("a web server is a keyboard device", 0, 4),
    ("a printer is a programming language", 0, 4),
    ("a motherboard is a text document format", 0, 4),
    ("a USB port is a neural network layer", 0, 4),


    # ========================================================
    # SCORE 3 — MODERATE
    # ========================================================

    ("a cache can reduce latency in some workloads", 1, 3),
    ("a queue can buffer work between software components", 1, 3),
    ("a load balancer can distribute requests among servers", 1, 3),
    ("compression can reduce the storage required for suitable data", 1, 3),
    ("a database index can speed up some queries", 1, 3),
    ("multithreading can improve throughput for some workloads", 1, 3),
    ("batch processing can be useful for large collections of data", 1, 3),
    ("logging can help with software troubleshooting", 1, 3),
    ("unit tests can detect some programming errors", 1, 3),
    ("a message queue can help decouple application components", 1, 3),
    ("replication can improve availability in some system designs", 1, 3),
    ("a compiler can report errors during compilation", 1, 3),

    ("caching always makes every application faster", 0, 3),
    ("a queue guarantees that every task completes successfully", 0, 3),
    ("a load balancer automatically fixes all server failures", 0, 3),
    ("compression always reduces every file to a smaller size", 0, 3),
    ("a database index makes every query instantaneous", 0, 3),
    ("multithreading always makes every program faster", 0, 3),
    ("batch processing requires every data item to be identical", 0, 3),
    ("logging automatically fixes software bugs", 0, 3),
    ("unit tests prove that software has no bugs", 0, 3),
    ("a message queue guarantees zero communication failures", 0, 3),
    ("replication guarantees that no server can ever fail", 0, 3),
    ("a compiler automatically fixes every source code error", 0, 3),


    # ========================================================
    # SCORE 2 — CONTEXT DEPENDENT
    # ========================================================

    ("a NoSQL database may suit applications with flexible data structures", 1, 2),
    ("microservices may suit some large application teams", 1, 2),
    ("serverless computing may reduce infrastructure management in some cases", 1, 2),
    ("containerization may simplify some deployment workflows", 1, 2),
    ("a larger memory allocation may help some memory-intensive workloads", 1, 2),
    ("parallel execution may improve performance for suitable algorithms", 1, 2),
    ("a content delivery network may reduce latency for some users", 1, 2),
    ("database replication may be useful when availability requirements are high", 1, 2),
    ("static typing may help detect some errors earlier", 1, 2),
    ("automated testing may reduce some maintenance risks", 1, 2),
    ("an SSD may improve storage-related application performance", 1, 2),
    ("a cloud platform may suit applications with variable workloads", 1, 2),

    ("NoSQL databases are always better than relational databases", 0, 2),
    ("microservices are always better than monolithic architecture", 0, 2),
    ("serverless computing always costs less than other architectures", 0, 2),
    ("containers always improve application performance", 0, 2),
    ("more memory always makes every program faster", 0, 2),
    ("parallel execution always improves every algorithm", 0, 2),
    ("a content delivery network always removes network latency", 0, 2),
    ("database replication always eliminates data consistency problems", 0, 2),
    ("static typing always prevents runtime errors", 0, 2),
    ("automated testing guarantees bug-free software", 0, 2),
    ("an SSD always makes every application run faster", 0, 2),
    ("cloud platforms always cost less than local infrastructure", 0, 2),


    # ========================================================
    # SCORE 1 — HIGHLY CONTEXT DEPENDENT
    # ========================================================

    ("whether a programming language is suitable depends on the task", 1, 1),
    ("whether a database design is suitable depends on workload requirements", 1, 1),
    ("whether a network architecture is appropriate depends on its requirements", 1, 1),
    ("whether a software tool is useful depends on the intended workflow", 1, 1),
    ("whether a storage technology is suitable depends on application needs", 1, 1),
    ("whether a deployment model is appropriate depends on operational constraints", 1, 1),
    ("whether an optimization is beneficial depends on the workload", 1, 1),
    ("whether automation is appropriate depends on the process being automated", 1, 1),
    ("whether replication is useful depends on system requirements", 1, 1),
    ("whether a programming architecture is appropriate depends on project constraints", 1, 1),
    ("whether a security control is sufficient depends on the threat model", 1, 1),
    ("whether a performance technique helps depends on the bottleneck", 1, 1),

    ("this architecture is guaranteed to be optimal for every application", 0, 1),
    ("this algorithm is guaranteed to produce the best possible result every time", 0, 1),
    ("this software system can never contain a defect", 0, 1),
    ("this security configuration guarantees complete protection against every threat", 0, 1),
    ("this database design is perfect for every possible workload", 0, 1),
    ("this hardware configuration is guaranteed to be ideal for every program", 0, 1),
    ("this optimization will always provide a performance improvement", 0, 1),
    ("this deployment model will always minimize operational cost", 0, 1),
    ("this backup strategy guarantees that no data can ever be lost", 0, 1),
    ("this machine learning model will always produce the correct answer", 0, 1),
    ("this network configuration can never fail", 0, 1),
    ("this software tool is guaranteed to be the best choice for everyone", 0, 1),
]


# ============================================================
# VERIFY CLAIM COUNT
# ============================================================

assert len(CLAIMS) == 120, (
    f"Expected 120 claims, got {len(CLAIMS)}"
)


# ============================================================
# VERIFY SCORE BALANCE
# ============================================================

score_counts = Counter(
    score
    for _, _, score in CLAIMS
)

yes_counts = Counter(
    yes
    for _, yes, _ in CLAIMS
)


for score in range(1, 6):

    assert score_counts[score] == 24, (
        f"Score {score} has "
        f"{score_counts[score]} claims; expected 24"
    )


assert yes_counts[0] == 60
assert yes_counts[1] == 60


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    return " ".join(
        text.split()
    )


# ============================================================
# CHECK CLAIM DUPLICATES
# ============================================================

normalized_claims = [
    normalize_text(
        claim
    )
    for claim, _, _ in CLAIMS
]

assert len(
    normalized_claims
) == len(
    set(normalized_claims)
), (
    "Duplicate claims detected."
)


# ============================================================
# LOAD ALL EVALUATION DATA
# ============================================================

evaluation_data = []

print("=" * 70)
print("DRISTI v0.4.1 TRAINING DATA PREPARATION")
print("=" * 70)

print()

for eval_file in EVAL_FILES:

    if not eval_file.exists():

        raise FileNotFoundError(
            f"Required evaluation file not found: "
            f"{eval_file}"
        )

    with open(
        eval_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    evaluation_data.extend(
        data
    )

    print(
        f"Loaded {len(data):>3} examples from "
        f"{eval_file}"
    )


print()

print(
    "Total evaluation examples loaded:",
    len(evaluation_data)
)


# ============================================================
# BUILD FORBIDDEN QUESTIONS
# ============================================================

forbidden_questions = {
    normalize_text(
        item["question"]
    )
    for item in evaluation_data
}


# ============================================================
# FILTER CLAIMS THAT OVERLAP WITH EVALUATION
# ============================================================

def claim_is_clean(claim):

    claim_key = normalize_text(
        claim
    )

    # --------------------------------------------------------
    # Check exact / substring overlap
    # --------------------------------------------------------

    for question in forbidden_questions:

        if (
            claim_key in question
            or question in claim_key
        ):

            return False

    # --------------------------------------------------------
    # Check generated question variants
    # --------------------------------------------------------

    for form in QUESTION_FORMS:

        generated_question = form.format(
            claim=claim
        )

        generated_key = normalize_text(
            generated_question
        )

        if generated_key in forbidden_questions:

            return False

    return True


clean_claims = [
    item
    for item in CLAIMS
    if claim_is_clean(
        item[0]
    )
]


# ============================================================
# REPORT FILTERING
# ============================================================

clean_score_counts = Counter(
    score
    for _, _, score in clean_claims
)

clean_yes_counts = Counter(
    yes
    for _, yes, _ in clean_claims
)

print()

print(
    "Clean claims remaining:",
    len(clean_claims)
)

print(
    "Clean TRUE claims:",
    clean_yes_counts[1]
)

print(
    "Clean FALSE claims:",
    clean_yes_counts[0]
)


# ============================================================
# REQUIRE ALL CLAIMS TO BE CLEAN
# ============================================================

if len(clean_claims) != 120:

    removed = [
        item
        for item in CLAIMS
        if item not in clean_claims
    ]

    print()

    print(
        "Claims rejected because they overlap "
        "with evaluation data:"
    )

    for claim, yes, score in removed:

        print(
            f"  score={score}, "
            f"yes={yes}: {claim}"
        )

    raise RuntimeError(
        "Training claims overlap with evaluation "
        "data. Need 120 clean claims."
    )


# ============================================================
# BUILD EXAMPLES
# ============================================================

examples = []

for claim, yes_label, score in clean_claims:

    for variant_index in range(
        VARIANTS_PER_CLAIM
    ):

        question = QUESTION_FORMS[
            variant_index
        ].format(
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
# VERIFY 1200 UNIQUE QUESTIONS
# ============================================================

assert len(examples) == 1200

unique_questions = {
    item["question"]
    for item in examples
}

assert len(unique_questions) == 1200, (
    "Training questions are not unique."
)


# ============================================================
# SPLIT TRUE / FALSE
# ============================================================

true_examples = [
    item
    for item in examples
    if item["yes"] == 1
]

false_examples = [
    item
    for item in examples
    if item["yes"] == 0
]


assert len(true_examples) == 600
assert len(false_examples) == 600


# ============================================================
# OPTION PERMUTATIONS
# ============================================================
#
# Four options:
#
# Yes
# No
# Maybe
# Unknown
#
# 4! = 24 permutations
#
# Each permutation appears 25 times for YES.
# Each permutation appears 25 times for NO.
#
# This produces 600 YES + 600 NO.
#
# ============================================================

all_permutations = list(
    permutations(OPTIONS)
)

assert len(all_permutations) == 24


def build_option_orders():

    orders = []

    for permutation_item in all_permutations:

        for _ in range(25):

            orders.append(
                list(permutation_item)
            )

    rng.shuffle(
        orders
    )

    return orders


yes_option_orders = build_option_orders()

no_option_orders = build_option_orders()


assert len(yes_option_orders) == 600
assert len(no_option_orders) == 600


# ============================================================
# BUILD FINAL DATASET
# ============================================================

dataset = []


for example, options in zip(
    true_examples,
    yes_option_orders
):

    correct_answer = "Yes"

    choice = options.index(
        correct_answer
    )

    dataset.append(
        {
            "question": example["question"],
            "options": options,
            "choice": choice,
            "yes": 1,
            "score": example["score"],
        }
    )


for example, options in zip(
    false_examples,
    no_option_orders
):

    correct_answer = "No"

    choice = options.index(
        correct_answer
    )

    dataset.append(
        {
            "question": example["question"],
            "options": options,
            "choice": choice,
            "yes": 0,
            "score": example["score"],
        }
    )


# ============================================================
# SHUFFLE FINAL DATASET
# ============================================================

rng.shuffle(
    dataset
)


# ============================================================
# FINAL COUNTS
# ============================================================

binary_counts = Counter(
    item["yes"]
    for item in dataset
)

score_counts = Counter(
    item["score"]
    for item in dataset
)

choice_counts = Counter(
    item["choice"]
    for item in dataset
)


# ============================================================
# OPTION POSITION COUNTS
# ============================================================

option_position_counts = {
    position: Counter()
    for position in range(4)
}


for item in dataset:

    for position, option in enumerate(
        item["options"]
    ):

        option_position_counts[position][
            option
        ] += 1


# ============================================================
# FINAL DATASET ASSERTIONS
# ============================================================

assert len(dataset) == 1200

assert len(
    {
        item["question"]
        for item in dataset
    }
) == 1200

assert len(
    {
        json.dumps(
            item,
            sort_keys=True
        )
        for item in dataset
    }
) == 1200


# ------------------------------------------------------------
# Binary balance
# ------------------------------------------------------------

assert binary_counts[0] == 600
assert binary_counts[1] == 600


# ------------------------------------------------------------
# Score balance
# ------------------------------------------------------------

for score in range(1, 6):

    assert score_counts[score] == 240


# ------------------------------------------------------------
# Choice balance
# ------------------------------------------------------------

for position in range(4):

    assert choice_counts[position] == 300


# ------------------------------------------------------------
# Option text balance
# ------------------------------------------------------------

for position in range(4):

    for option in OPTIONS:

        count = option_position_counts[
            position
        ][option]

        assert count == 300, (
            f"{option} occurs {count} times "
            f"at position {position}; "
            f"expected 300."
        )


# ============================================================
# CROSS-DATASET OVERLAP CHECK
# ============================================================

training_questions = {
    normalize_text(
        item["question"]
    )
    for item in dataset
}


overlap_results = {}


for eval_file in EVAL_FILES:

    with open(
        eval_file,
        "r",
        encoding="utf-8"
    ) as file:

        eval_data = json.load(
            file
        )

    eval_questions = {
        normalize_text(
            item["question"]
        )
        for item in eval_data
    }

    overlap = (
        training_questions
        & eval_questions
    )

    overlap_results[
        str(eval_file)
    ] = len(overlap)

    assert len(overlap) == 0, (
        f"Overlap found with {eval_file}: "
        f"{len(overlap)} questions"
    )


# ============================================================
# BACKUP CURRENT TRAIN.JSON
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

if OUTPUT_FILE.exists():

    backup_path = BACKUP_FILE

    counter = 2

    while backup_path.exists():

        backup_path = Path(
            f"data/train_v041_backup_{counter}.json"
        )

        counter += 1

    shutil.copy2(
        OUTPUT_FILE,
        backup_path
    )

    print()

    print(
        "Previous train.json backed up to:"
    )

    print(
        backup_path
    )


# ============================================================
# SAVE NEW TRAINING DATA
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        dataset,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("DRISTI v0.4.1 TRAINING DATASET CREATED")
print("=" * 70)

print()

print(
    "Total examples:",
    len(dataset)
)

print()

print("Binary labels:")

print(
    f"  NO  : {binary_counts[0]}"
)

print(
    f"  YES : {binary_counts[1]}"
)

print()

print("Score distribution:")

for score in range(1, 6):

    print(
        f"  Score {score}: "
        f"{score_counts[score]}"
    )

print()

print("Correct choice positions:")

for position in range(4):

    print(
        f"  Position {position}: "
        f"{choice_counts[position]}"
    )

print()

print(
    "Option text distribution by position:"
)

for position in range(4):

    print(
        f"  Position {position}: "
        f"{dict(option_position_counts[position])}"
    )

print()

print(
    "Unique questions:",
    len(unique_questions)
)

print(
    "Unique complete examples:",
    len(
        {
            json.dumps(
                item,
                sort_keys=True
            )
            for item in dataset
        }
    )
)

print()

print(
    "Evaluation overlap:"
)

for eval_file in EVAL_FILES:

    print(
        f"  {eval_file}: "
        f"{overlap_results[str(eval_file)]}"
    )

print()

print(
    "Saved to:",
    OUTPUT_FILE
)

print()

print("=" * 70)
print(
    "DRISTI v0.4.1 TRAINING DATA COMPLETE"
)
print("=" * 70)