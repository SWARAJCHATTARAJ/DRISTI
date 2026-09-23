import json
import random
from pathlib import Path


random.seed(12345)


# =========================================================
# Each set has:
#
# 20 unique templates
# 5 variants per template
# = 100 examples
#
# Each set contains:
# YES = 50
# NO  = 50
#
# Scores:
# 1 = 20
# 2 = 20
# 3 = 20
# 4 = 20
# 5 = 20
#
# IMPORTANT:
# No base template is shared between validation,
# calibration, and test.
# =========================================================


VALIDATION_TEMPLATES = [

    # SCORE 1
    (
        "JPEG is a programming language.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 1
    ),
    (
        "PNG is a database system.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 1
    ),
    (
        "Python is a programming language.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 1
    ),
    (
        "PostgreSQL is a relational database.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 1
    ),

    # SCORE 2
    (
        "CSS is normally used to train neural networks.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 2
    ),
    (
        "Photoshop is a database engine.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 2
    ),
    (
        "Python can be used for automation.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 2
    ),
    (
        "A database can store customer records.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 2
    ),

    # SCORE 3
    (
        "HTML is a GPU programming framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 3
    ),
    (
        "JPEG is normally used as a database engine.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 3
    ),
    (
        "Python is useful for data analysis.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 3
    ),
    (
        "A CPU can perform numerical calculations.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 3
    ),

    # SCORE 4
    (
        "CSS is a relational database.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 4
    ),
    (
        "MP3 is a programming language.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 4
    ),
    (
        "PyTorch can build neural networks.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 4
    ),
    (
        "CUDA is associated with NVIDIA GPU computing.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 4
    ),

    # SCORE 5
    (
        "HTML is a machine learning framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 5
    ),
    (
        "GIF is a relational database.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 5
    ),
    (
        "TensorFlow is used for machine learning.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 5
    ),
    (
        "SQL can be used to query relational databases.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 5
    ),
]


CALIBRATION_TEMPLATES = [

    # SCORE 1
    (
        "BMP is a programming language.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 1
    ),
    (
        "WAV is a relational database.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 1
    ),
    (
        "Java is a programming language.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 1
    ),
    (
        "SQLite is a database engine.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 1
    ),

    # SCORE 2
    (
        "CSS is a database query language.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 2
    ),
    (
        "PowerPoint is a neural network framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 2
    ),
    (
        "JavaScript can be used for web development.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 2
    ),
    (
        "A CPU can execute instructions.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 2
    ),

    # SCORE 3
    (
        "PDF is a machine learning framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 3
    ),
    (
        "MP4 is a database management system.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 3
    ),
    (
        "NumPy is used for numerical computing.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 3
    ),
    (
        "A neural network can process numerical inputs.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 3
    ),

    # SCORE 4
    (
        "Notepad is a deep learning framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 4
    ),
    (
        "PNG is a programming environment.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 4
    ),
    (
        "Pandas is commonly used for data manipulation.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 4
    ),
    (
        "Git is commonly used for version control.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 4
    ),

    # SCORE 5
    (
        "JPEG is a programming framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 5
    ),
    (
        "Excel is a relational database engine.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 5
    ),
    (
        "FastAPI can be used to build web APIs.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 5
    ),
    (
        "Linux is an operating system.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 5
    ),
]


TEST_TEMPLATES = [

    # SCORE 1
    (
        "TIFF is a programming language.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 1
    ),
    (
        "AVI is a relational database.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 1
    ),
    (
        "C++ is a programming language.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 1
    ),
    (
        "MySQL is a database system.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 1
    ),

    # SCORE 2
    (
        "HTML is normally used for GPU programming.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 2
    ),
    (
        "Photoshop is a relational database.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 2
    ),
    (
        "Ruby can be used for software development.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 2
    ),
    (
        "An operating system manages computer resources.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 2
    ),

    # SCORE 3
    (
        "SVG is a database engine.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 3
    ),
    (
        "DOCX is a machine learning framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 3
    ),
    (
        "Scikit-learn provides machine learning tools.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 3
    ),
    (
        "A transformer model can process tokenized text.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 3
    ),

    # SCORE 4
    (
        "MP3 is a database management system.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 4
    ),
    (
        "PowerPoint is a GPU computing framework.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 4
    ),
    (
        "JSON is commonly used for structured data exchange.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 4
    ),
    (
        "Git can track changes to source code.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 4
    ),

    # SCORE 5
    (
        "PNG is a neural network architecture.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 5
    ),
    (
        "Word is a GPU programming platform.",
        ["Yes", "No", "Maybe", "Unknown"],
        1, 0, 5
    ),
    (
        "NumPy can perform numerical array operations.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 5
    ),
    (
        "Docker can package applications into containers.",
        ["Yes", "No", "Maybe", "Unknown"],
        0, 1, 5
    ),
]


# =========================================================
# CREATE VARIANTS
# =========================================================

def create_variants(
    question,
    options,
    correct_index,
    yes_label,
    score
):

    variants = [
        question,

        "Fact check: " + question,

        "Is it true that "
        + question.lower(),

        "Can we say that "
        + question.lower(),

        "Consider this statement: "
        + question
    ]

    results = []

    for text in variants:

        shuffled = options.copy()

        correct_answer = shuffled[
            correct_index
        ]

        random.shuffle(shuffled)

        new_choice = shuffled.index(
            correct_answer
        )

        results.append({
            "question": text,
            "options": shuffled,
            "choice": new_choice,
            "yes": yes_label,
            "score": score
        })

    return results


# =========================================================
# GENERATE DATASET
# =========================================================

def generate_dataset(templates):

    dataset = []

    for template in templates:

        (
            question,
            options,
            choice,
            yes,
            score
        ) = template

        dataset.extend(
            create_variants(
                question,
                options,
                choice,
                yes,
                score
            )
        )

    random.shuffle(dataset)

    return dataset


# =========================================================
# SAVE
# =========================================================

def save_dataset(
    filename,
    dataset
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False
        )


Path("data").mkdir(
    parents=True,
    exist_ok=True
)


validation = generate_dataset(
    VALIDATION_TEMPLATES
)

calibration = generate_dataset(
    CALIBRATION_TEMPLATES
)

test = generate_dataset(
    TEST_TEMPLATES
)


save_dataset(
    "data/validation.json",
    validation
)

save_dataset(
    "data/calibration.json",
    calibration
)

save_dataset(
    "data/test.json",
    test
)


# =========================================================
# STATISTICS
# =========================================================

def stats(dataset):

    yes = sum(
        x["yes"] == 1
        for x in dataset
    )

    no = sum(
        x["yes"] == 0
        for x in dataset
    )

    scores = {}

    choices = {}

    for item in dataset:

        score = item["score"]

        scores[score] = (
            scores.get(score, 0) + 1
        )

        choice = item["choice"]

        choices[choice] = (
            choices.get(choice, 0) + 1
        )

    return yes, no, scores, choices


# =========================================================
# DISPLAY
# =========================================================

print("=" * 60)
print("       DRISTI INDEPENDENT EVALUATION DATA")
print("=" * 60)


for name, dataset in [
    ("VALIDATION", validation),
    ("CALIBRATION", calibration),
    ("TEST", test)
]:

    yes, no, scores, choices = stats(
        dataset
    )

    print()
    print(name)
    print("-" * 40)

    print(
        "Examples:",
        len(dataset)
    )

    print(
        "YES:",
        yes
    )

    print(
        "NO:",
        no
    )

    print(
        "Scores:",
        dict(sorted(scores.items()))
    )

    print(
        "Choices:",
        dict(sorted(choices.items()))
    )


print()
print("Created:")
print("data/validation.json")
print("data/calibration.json")
print("data/test.json")

print("=" * 60)