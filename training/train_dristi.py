import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.dristi_model import DristiModel
from training.dristi_dataset import DristiDataset


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_FILE = "data/train.json"

BATCH_SIZE = 1

EPOCHS = 5

LEARNING_RATE = 2e-5

MAX_LENGTH = 128


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("                 DRISTI TRAINING")
    print("=" * 60)

    print("PyTorch version:", torch.__version__)
    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    print("=" * 60)


    # =====================================================
    # DATASET
    # =====================================================

    dataset = DristiDataset(
        TRAIN_FILE,
        max_length=MAX_LENGTH
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    print("Training examples:", len(dataset))


    # =====================================================
    # MODEL
    # =====================================================

    model = DristiModel(
        num_scores=5
    )

    model.to(DEVICE)


    # Gradient checkpointing reduces VRAM usage.
    if torch.cuda.is_available():

        model.encoder.gradient_checkpointing_enable()

        print("Gradient checkpointing: ENABLED")


    # =====================================================
    # LOSS FUNCTIONS
    # =====================================================

    binary_loss_fn = nn.BCEWithLogitsLoss()

    choice_loss_fn = nn.CrossEntropyLoss()

    score_loss_fn = nn.CrossEntropyLoss()


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
    # TRAINING LOOP
    # =====================================================

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        print()
        print(
            f"Starting Epoch {epoch + 1}/{EPOCHS}"
        )

        for batch_number, batch in enumerate(loader):

            # ---------------------------------------------
            # Move question tensors to GPU
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
            # Move option tensors to GPU
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

            # ---------------------------------------------
            # We currently use batch size = 1.
            # Remove that extra dimension from options.
            # ---------------------------------------------

            option_input_ids = option_input_ids[0]

            option_attention_mask = (
                option_attention_mask[0]
            )

            # ---------------------------------------------
            # Reset gradients
            # ---------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------
            # Forward pass
            # ---------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                # Question → binary + score
                question_output = (
                    model.forward_question(
                        question_input_ids,
                        question_attention_mask
                    )
                )

                # Question + each option → option scores
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
                        question_output["binary"],
                        yes_labels
                    )
                )

                # -----------------------------------------
                # Loss 2: Dynamic choice
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
                        question_output["score"],
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

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()

            print(
                f"  Batch {batch_number + 1}/"
                f"{len(loader)} "
                f"| Loss: {loss.item():.4f}"
            )


        # =================================================
        # EPOCH RESULT
        # =================================================

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1} completed "
            f"| Average Loss: "
            f"{average_loss:.4f}"
        )

        # Free unused CUDA memory
        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # =====================================================
    # SAVE MODEL
    # =====================================================

    save_path = (
        "checkpoints/dristi_model.pt"
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print()
    print("=" * 60)
    print("DRISTI TRAINING COMPLETED")
    print("=" * 60)
    print("Model saved to:")
    print(save_path)
    print("=" * 60)


if __name__ == "__main__":
    main()