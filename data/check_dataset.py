import json
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================

FILES = {
    "TRAIN": "data/train.json",
    "VALIDATION": "data/validation.json",
    "CALIBRATION": "data/calibration.json",
    "TEST": "data/test.json",
}


# ============================================================
# LOAD JSON
# ============================================================

def load_data(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# CHECK BASIC STRUCTURE
# ============================================================

def check_required_fields(data, dataset_name):

    required_fields = [
        "question",
        "options",
        "choice",
        "yes",
        "score",
    ]

    missing = []

    for index, item in enumerate(data):

        for field in required_fields:

            if field not in item:

                missing.append(
                    (index, field)
                )

    print(
        f"{dataset_name} missing fields: "
        f"{len(missing)}"
    )

    if missing:

        for index, field in missing[:10]:

            print(
                f"  Index {index}: missing '{field}'"
            )


# ============================================================
# CHECK LABEL RANGES
# ============================================================

def check_label_ranges(data, dataset_name):

    bad_yes = []
    bad_choice = []
    bad_score = []
    bad_options = []

    for index, item in enumerate(data):

        yes = item.get("yes")
        choice = item.get("choice")
        score = item.get("score")
        options = item.get("options", [])

        if yes not in [0, 1]:

            bad_yes.append(
                (index, yes)
            )

        if not isinstance(choice, int) or not (
            0 <= choice < len(options)
        ):

            bad_choice.append(
                (index, choice, len(options))
            )

        if score not in [1, 2, 3, 4, 5]:

            bad_score.append(
                (index, score)
            )

        if len(options) != 4:

            bad_options.append(
                (index, len(options))
            )

    print(
        f"{dataset_name} bad YES labels: "
        f"{len(bad_yes)}"
    )

    print(
        f"{dataset_name} bad choice labels: "
        f"{len(bad_choice)}"
    )

    print(
        f"{dataset_name} bad score labels: "
        f"{len(bad_score)}"
    )

    print(
        f"{dataset_name} non-4-option examples: "
        f"{len(bad_options)}"
    )


# ============================================================
# CHECK QUESTION / OPTION RELATIONSHIPS
# ============================================================

def check_semantic_choice_consistency(
    data,
    dataset_name
):

    suspicious = []

    for index, item in enumerate(data):

        question = item["question"].lower()

        options = [
            str(option).strip().lower()
            for option in item["options"]
        ]

        choice = item["choice"]

        selected_option = options[choice]

        yes = item["yes"]

        # ----------------------------------------------------
        # Basic YES / NO consistency
        # ----------------------------------------------------

        contains_yes = "yes" in selected_option
        contains_no = (
            selected_option == "no"
            or selected_option.startswith("no ")
        )

        if yes == 1 and not contains_yes:

            suspicious.append({
                "index": index,
                "reason": "YES label but selected option is not YES",
                "question": item["question"],
                "selected_option": selected_option,
                "yes": yes,
                "score": item["score"],
            })

        if yes == 0 and contains_yes:

            suspicious.append({
                "index": index,
                "reason": "NO label but selected option is YES",
                "question": item["question"],
                "selected_option": selected_option,
                "yes": yes,
                "score": item["score"],
            })

        # ----------------------------------------------------
        # Print if question contains obvious fact words
        # ----------------------------------------------------

        if (
            "true that" in question
            or "statement" in question
            or "fact check" in question
        ):
            pass

    print(
        f"{dataset_name} suspicious YES/NO-choice pairs: "
        f"{len(suspicious)}"
    )

    for item in suspicious[:10]:

        print()

        print(
            f"Index      : {item['index']}"
        )

        print(
            f"Reason     : {item['reason']}"
        )

        print(
            f"Question   : {item['question']}"
        )

        print(
            f"Selected   : {item['selected_option']}"
        )

        print(
            f"YES label  : {item['yes']}"
        )

        print(
            f"Score      : {item['score']}"
        )


# ============================================================
# DISTRIBUTIONS
# ============================================================

def print_distributions(
    data,
    dataset_name
):

    yes_counts = Counter()

    choice_counts = Counter()

    score_counts = Counter()

    option_position_text = defaultdict(Counter)

    for item in data:

        yes_counts[item["yes"]] += 1

        choice_counts[item["choice"]] += 1

        score_counts[item["score"]] += 1

        for position, option in enumerate(
            item["options"]
        ):

            option_position_text[position][
                str(option).strip().lower()
            ] += 1

    print()
    print(
        f"{dataset_name} distributions"
    )

    print(
        "  YES:"
        f" NO={yes_counts[0]}"
        f", YES={yes_counts[1]}"
    )

    print(
        "  Choice:"
    )

    for position in range(4):

        print(
            f"    Option {position + 1}: "
            f"{choice_counts[position]}"
        )

    print(
        "  Score:"
    )

    for score in range(1, 6):

        print(
            f"    {score}: "
            f"{score_counts[score]}"
        )

    print()
    print(
        "  Selected option text by position:"
    )

    for position in range(4):

        print(
            f"    Position {position + 1}: "
            f"{dict(option_position_text[position])}"
        )


# ============================================================
# CHECK DUPLICATES
# ============================================================

def check_duplicates(
    data,
    dataset_name
):

    question_counter = Counter()

    exact_counter = Counter()

    for item in data:

        question = (
            item["question"]
            .strip()
            .lower()
        )

        exact = json.dumps(
            item,
            sort_keys=True
        )

        question_counter[question] += 1

        exact_counter[exact] += 1

    duplicate_questions = {
        q: count
        for q, count in question_counter.items()
        if count > 1
    }

    exact_duplicates = {
        q: count
        for q, count in exact_counter.items()
        if count > 1
    }

    print()

    print(
        f"{dataset_name} duplicate questions: "
        f"{len(duplicate_questions)}"
    )

    print(
        f"{dataset_name} exact duplicate examples: "
        f"{len(exact_duplicates)}"
    )

    if duplicate_questions:

        print(
            "  Sample duplicate questions:"
        )

        for question, count in list(
            duplicate_questions.items()
        )[:5]:

            print(
                f"    ({count}x) {question}"
            )


# ============================================================
# CROSS-DATASET OVERLAP
# ============================================================

def check_cross_dataset_overlap(
    loaded
):

    print()
    print("=" * 70)
    print("CROSS-DATASET QUESTION OVERLAP")
    print("=" * 70)

    question_sets = {}

    for name, data in loaded.items():

        question_sets[name] = {
            item["question"].strip().lower()
            for item in data
        }

    names = list(question_sets.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            name_a = names[i]
            name_b = names[j]

            overlap = (
                question_sets[name_a]
                & question_sets[name_b]
            )

            print(
                f"{name_a} ↔ {name_b}: "
                f"{len(overlap)} overlapping questions"
            )

            if overlap:

                for question in list(overlap)[:3]:

                    print(
                        f"  {question}"
                    )


# ============================================================
# LABEL RELATIONSHIP TABLE
# ============================================================

def print_label_relationship(
    data,
    dataset_name
):

    relationship = Counter()

    for item in data:

        relationship[
            (
                item["yes"],
                item["choice"],
                item["score"]
            )
        ] += 1

    print()
    print(
        f"{dataset_name} label relationships"
    )

    print(
        "  Format: YES, CHOICE, SCORE -> COUNT"
    )

    for (
        yes,
        choice,
        score
    ), count in sorted(
        relationship.items()
    ):

        print(
            f"  ({yes}, {choice}, {score}) "
            f"-> {count}"
        )


# ============================================================
# SAMPLE DATA
# ============================================================

def print_samples(
    data,
    dataset_name
):

    print()
    print(
        f"{dataset_name} first 5 examples"
    )

    for index, item in enumerate(
        data[:5]
    ):

        print()
        print(
            f"Example {index}"
        )

        print(
            f"Question: {item['question']}"
        )

        print("Options:")

        for position, option in enumerate(
            item["options"]
        ):

            marker = (
                "<-- SELECTED"
                if position == item["choice"]
                else ""
            )

            print(
                f"  {position}: "
                f"{option} {marker}"
            )

        print(
            f"YES   : {item['yes']}"
        )

        print(
            f"CHOICE: {item['choice']}"
        )

        print(
            f"SCORE : {item['score']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("        DRISTI DATASET CONSISTENCY CHECK")
    print("=" * 70)

    loaded = {}

    # --------------------------------------------------------
    # Load all datasets
    # --------------------------------------------------------

    for name, path in FILES.items():

        print()
        print(
            f"Loading {name}: {path}"
        )

        try:

            data = load_data(path)

            loaded[name] = data

            print(
                f"  Loaded: {len(data)} examples"
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

    # --------------------------------------------------------
    # Analyze each dataset
    # --------------------------------------------------------

    for name, data in loaded.items():

        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        check_required_fields(
            data,
            name
        )

        check_label_ranges(
            data,
            name
        )

        check_semantic_choice_consistency(
            data,
            name
        )

        print_distributions(
            data,
            name
        )

        check_duplicates(
            data,
            name
        )

        print_label_relationship(
            data,
            name
        )

        print_samples(
            data,
            name
        )

    # --------------------------------------------------------
    # Cross dataset overlap
    # --------------------------------------------------------

    if loaded:

        check_cross_dataset_overlap(
            loaded
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DRISTI DATASET CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()