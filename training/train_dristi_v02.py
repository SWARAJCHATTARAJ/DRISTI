import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.dristi_model import DristiModel
from training.dristi_dataset import DristiDataset


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_FILE = "data/train.json"

VALIDATION_FILE = "data/validation.json"

CHECKPOINT_FILE = (
    "checkpoints/dristi_v02_best.pt"
)

EPOCHS = 5

BATCH_SIZE = 1

LEARNING_RATE = 2e-5

MAX_LENGTH = 128


# =========================================================
# VALIDATION FUNCTION
# =========================================================

def validate(
    model,
    loader,
    binary_loss_fn,
    choice_loss_fn,
    score_loss_fn
):

    model.eval()

    total_loss = 0.0

    total_binary_loss = 0.0

    total_choice_loss = 0.0

    total_score_loss = 0.0

    correct_choices = 0

    total_examples = 0


    with torch.no_grad():

        for batch in loader:

            question_input_ids = (
                batch["question_input_ids"]
                .to(DEVICE)
            )

            question_attention_mask = (
                batch["question_attention_mask"]
                .to(DEVICE)
            )

            option_input_ids = (
                batch["option_input_ids"]
                .to(DEVICE)
            )

            option_attention_mask = (
                batch["option_attention_mask"]
                .to(DEVICE)
            )

            choice_labels = (
                batch["choice"]
                .to(DEVICE)
            )

            yes_labels = (
                batch["yes"]
                .to(DEVICE)
            )

            score_labels = (
                batch["score"]
                .to(DEVICE)
            )


            # Remove batch dimension.
            option_input_ids = (
                option_input_ids[0]
            )

            option_attention_mask = (
                option_attention_mask[0]
            )


            # -----------------------------------------
            # Forward pass
            # -----------------------------------------

            outputs = model.forward_question(
                question_input_ids,
                question_attention_mask
            )

            option_logits = (
                model.score_options(
                    option_input_ids,
                    option_attention_mask
                )
            )


            # -----------------------------------------
            # Losses
            # -----------------------------------------

            binary_loss = binary_loss_fn(
                outputs["binary"],
                yes_labels
            )

            choice_loss = choice_loss_fn(
                option_logits.unsqueeze(0),
                choice_labels
            )

            score_loss = score_loss_fn(
                outputs["score"],
                score_labels
            )


            loss = (
                binary_loss
                + choice_loss
                + (0.5 * score_loss)
            )


            total_loss += loss.item()

            total_binary_loss += (
                binary_loss.item()
            )

            total_choice_loss += (
                choice_loss.item()
            )

            total_score_loss += (
                score_loss.item()
            )


            # -----------------------------------------
            # Choice accuracy
            # -----------------------------------------

            prediction = torch.argmax(
                option_logits
            ).item()

            correct = (
                prediction
                == choice_labels.item()
            )

            if correct:
                correct_choices += 1

            total_examples += 1


    average_loss = (
        total_loss / total_examples
    )

    average_binary_loss = (
        total_binary_loss
        / total_examples
    )

    average_choice_loss = (
        total_choice_loss
        / total_examples
    )

    average_score_loss = (
        total_score_loss
        / total_examples
    )

    choice_accuracy = (
        correct_choices
        / total_examples
    )


    return {
        "loss": average_loss,
        "binary_loss": average_binary_loss,
        "choice_loss": average_choice_loss,
        "score_loss": average_score_loss,
        "choice_accuracy": choice_accuracy
    }


# =========================================================
# MAIN TRAINING
# =========================================================

def main():

    print("=" * 65)
    print("                     DRISTI v0.2")
    print("                 TRAINING PIPELINE")
    print("=" * 65)

    print("PyTorch:", torch.__version__)

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    print("=" * 65)


    # =====================================================
    # LOAD DATASETS
    # =====================================================

    train_dataset = DristiDataset(
        TRAIN_FILE,
        max_length=MAX_LENGTH
    )

    validation_dataset = DristiDataset(
        VALIDATION_FILE,
        max_length=MAX_LENGTH
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )


    print(
        "Training examples:",
        len(train_dataset)
    )

    print(
        "Validation examples:",
        len(validation_dataset)
    )


    # =====================================================
    # MODEL
    # =====================================================

    model = DristiModel(
        num_scores=5
    )

    model.to(DEVICE)


    # =====================================================
    # MEMORY OPTIMIZATION
    # =====================================================

    if torch.cuda.is_available():

        model.encoder.gradient_checkpointing_enable()

        print(
            "Gradient checkpointing: ENABLED"
        )


    # =====================================================
    # BINARY CLASS WEIGHT
    # =====================================================

    # Training set:
    #
    # YES = 800
    # NO  = 400
    #
    # BCEWithLogitsLoss with pos_weight=0.5
    # reduces the effect of the more frequent YES class.

    POS_WEIGHT = torch.tensor(
        [0.5],
        device=DEVICE
    )


    binary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=POS_WEIGHT
    )


    choice_loss_fn = (
        nn.CrossEntropyLoss()
    )


    score_loss_fn = (
        nn.CrossEntropyLoss()
    )


    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # =====================================================
    # MIXED PRECISION
    # =====================================================

    use_amp = torch.cuda.is_available()


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )


    # =====================================================
    # BEST MODEL TRACKING
    # =====================================================

    best_validation_loss = float(
        "inf"
    )


    # =====================================================
    # TRAIN
    # =====================================================

    for epoch in range(EPOCHS):

        model.train()

        total_training_loss = 0.0


        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("-" * 65)


        for batch_number, batch in enumerate(
            train_loader
        ):

            # ---------------------------------------------
            # Question
            # ---------------------------------------------

            question_input_ids = (
                batch["question_input_ids"]
                .to(DEVICE)
            )

            question_attention_mask = (
                batch["question_attention_mask"]
                .to(DEVICE)
            )


            # ---------------------------------------------
            # Options
            # ---------------------------------------------

            option_input_ids = (
                batch["option_input_ids"]
                .to(DEVICE)
            )

            option_attention_mask = (
                batch["option_attention_mask"]
                .to(DEVICE)
            )


            # ---------------------------------------------
            # Labels
            # ---------------------------------------------

            choice_labels = (
                batch["choice"]
                .to(DEVICE)
            )

            yes_labels = (
                batch["yes"]
                .to(DEVICE)
            )

            score_labels = (
                batch["score"]
                .to(DEVICE)
            )


            # Batch size = 1
            option_input_ids = (
                option_input_ids[0]
            )

            option_attention_mask = (
                option_attention_mask[0]
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                question_outputs = (
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


                # -----------------------------------------
                # Loss 1: Binary
                # -----------------------------------------

                binary_loss = (
                    binary_loss_fn(
                        question_outputs["binary"],
                        yes_labels
                    )
                )


                # -----------------------------------------
                # Loss 2: Choice
                # -----------------------------------------

                choice_loss = (
                    choice_loss_fn(
                        option_logits.unsqueeze(0),
                        choice_labels
                    )
                )


                # -----------------------------------------
                # Loss 3: Score
                # -----------------------------------------

                score_loss = (
                    score_loss_fn(
                        question_outputs["score"],
                        score_labels
                    )
                )


                # -----------------------------------------
                # Total loss
                # -----------------------------------------

                loss = (
                    binary_loss
                    + choice_loss
                    + (0.5 * score_loss)
                )


            # ---------------------------------------------
            # Backpropagation
            # ---------------------------------------------

            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            total_training_loss += (
                loss.item()
            )


            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if (
                batch_number + 1
            ) % 100 == 0:

                print(
                    f"Batch "
                    f"{batch_number + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{loss.item():.4f}"
                )


        # =================================================
        # TRAINING LOSS
        # =================================================

        average_training_loss = (
            total_training_loss
            / len(train_loader)
        )


        # =================================================
        # VALIDATION
        # =================================================

        metrics = validate(
            model,
            validation_loader,
            binary_loss_fn,
            choice_loss_fn,
            score_loss_fn
        )


        print()
        print(
            "Training loss:",
            f"{average_training_loss:.4f}"
        )

        print(
            "Validation loss:",
            f"{metrics['loss']:.4f}"
        )

        print(
            "Validation binary loss:",
            f"{metrics['binary_loss']:.4f}"
        )

        print(
            "Validation choice loss:",
            f"{metrics['choice_loss']:.4f}"
        )

        print(
            "Validation score loss:",
            f"{metrics['score_loss']:.4f}"
        )

        print(
            "Validation choice accuracy:",
            f"{metrics['choice_accuracy']:.4f}"
        )


        # =================================================
        # SAVE BEST MODEL
        # =================================================

        if (
            metrics["loss"]
            < best_validation_loss
        ):

            best_validation_loss = (
                metrics["loss"]
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch + 1,

                    "validation_loss":
                        metrics["loss"],

                    "choice_accuracy":
                        metrics[
                            "choice_accuracy"
                        ],

                    "learning_rate":
                        LEARNING_RATE
                },
                CHECKPOINT_FILE
            )


            print()
            print(
                "✅ New best model saved:"
            )

            print(
                CHECKPOINT_FILE
            )


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # =====================================================
    # COMPLETE
    # =====================================================

    print()
    print("=" * 65)
    print("              DRISTI v0.2 TRAINING COMPLETE")
    print("=" * 65)

    print(
        "Best validation loss:",
        f"{best_validation_loss:.4f}"
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_FILE
    )

    print("=" * 65)


if __name__ == "__main__":

    main()