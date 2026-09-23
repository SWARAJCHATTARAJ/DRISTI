import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dristi_model_v03 import DristiModelV03
from training.dristi_dataset import DristiDataset


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

TRAIN_FILE = "data/train.json"

# IMPORTANT:
# v0.4.1 uses the new evaluation set.
VAL_FILE = "data/validation_v041.json"

CHECKPOINT_DIR = "checkpoints"

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "dristi_v041_best.pt"
)

BATCH_SIZE = 2
EPOCHS = 6

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

MAX_LENGTH = 128

PATIENCE = 2

ORDINAL_WEIGHT = 0.5

SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("              DRISTI v0.4.1 TRAINING")
print("=" * 70)

print(
    f"Device: {device}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("=" * 70)


# ============================================================
# DATASETS
# ============================================================

print()
print("Loading datasets...")


train_dataset = DristiDataset(
    TRAIN_FILE,
    max_length=MAX_LENGTH
)


val_dataset = DristiDataset(
    VAL_FILE,
    max_length=MAX_LENGTH
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    f"Training examples  : {len(train_dataset)}"
)

print(
    f"Validation examples: {len(val_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading model...")


model = DristiModelV03(
    model_name=MODEL_NAME,
    num_scores=5
)


# ------------------------------------------------------------
# Gradient checkpointing
# ------------------------------------------------------------

try:

    model.encoder.gradient_checkpointing_enable()

    print(
        "Gradient checkpointing: ON"
    )

except Exception:

    print(
        "Gradient checkpointing: OFF"
    )


model = model.to(device)


print(
    "Model loaded successfully."
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# Binary classification
#
# v0.4.1 training data is balanced:
#
# NO  = 600
# YES = 600
#
# Therefore no binary class weighting is required.
# ------------------------------------------------------------

binary_loss_fn = nn.CrossEntropyLoss()


# ------------------------------------------------------------
# Dynamic choice
# ------------------------------------------------------------

choice_loss_fn = nn.CrossEntropyLoss()


# ============================================================
# ORDINAL LOSS
# ============================================================
#
# Dataset stores:
#
# score 1 -> label 0
# score 2 -> label 1
# score 3 -> label 2
# score 4 -> label 3
# score 5 -> label 4
#
# Ordinal targets:
#
# score 1 -> [0,0,0,0]
# score 2 -> [1,0,0,0]
# score 3 -> [1,1,0,0]
# score 4 -> [1,1,1,0]
# score 5 -> [1,1,1,1]
#
# ============================================================

class OrdinalLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.register_buffer(
            "pos_weight",
            torch.tensor(
                [
                    0.25,
                    2.0 / 3.0,
                    1.5,
                    4.0
                ],
                dtype=torch.float32
            )
        )


    def forward(
        self,
        logits,
        scores
    ):

        thresholds = torch.arange(
            1,
            5,
            device=scores.device
        ).unsqueeze(0)


        targets = (
            scores.unsqueeze(1)
            >= thresholds
        ).float()


        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )


        return loss


ordinal_loss_fn = OrdinalLoss().to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True
    )

else:

    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=False
    )


# ============================================================
# ORDINAL EXPECTED SCORE
# ============================================================

def ordinal_expected_score(
    ordinal_logits
):

    probabilities = torch.sigmoid(
        ordinal_logits
    )


    expected_score = (
        1.0
        + probabilities.sum(
            dim=1
        )
    )


    return expected_score


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()


    total_loss = 0.0

    total_binary_loss = 0.0

    total_choice_loss = 0.0

    total_ordinal_loss = 0.0

    num_batches = 0


    for batch in train_loader:

        question_input_ids = batch[
            "question_input_ids"
        ].to(device)


        question_attention_mask = batch[
            "question_attention_mask"
        ].to(device)


        option_input_ids = batch[
            "option_input_ids"
        ].to(device)


        option_attention_mask = batch[
            "option_attention_mask"
        ].to(device)


        choice_labels = batch[
            "choice"
        ].to(device)


        binary_labels = batch[
            "yes"
        ].long().to(device)


        score_labels = batch[
            "score"
        ].to(device)


        optimizer.zero_grad(
            set_to_none=True
        )


        if use_amp:

            autocast_context = torch.amp.autocast(
                "cuda",
                enabled=True
            )

        else:

            autocast_context = torch.amp.autocast(
                "cpu",
                enabled=False
            )


        with autocast_context:

            # ------------------------------------------------
            # Question outputs
            # ------------------------------------------------

            question_output = model.forward_question(
                question_input_ids,
                question_attention_mask
            )


            # ------------------------------------------------
            # Option outputs
            # ------------------------------------------------

            option_logits = model.score_options(
                option_input_ids,
                option_attention_mask
            )


            binary_logits = question_output[
                "binary"
            ]


            ordinal_logits = question_output[
                "ordinal"
            ]


            # ------------------------------------------------
            # Losses
            # ------------------------------------------------

            binary_loss = binary_loss_fn(
                binary_logits,
                binary_labels
            )


            choice_loss = choice_loss_fn(
                option_logits,
                choice_labels
            )


            ordinal_loss = ordinal_loss_fn(
                ordinal_logits,
                score_labels
            )


            # ------------------------------------------------
            # Total loss
            # ------------------------------------------------

            total_batch_loss = (
                binary_loss
                + choice_loss
                + ORDINAL_WEIGHT * ordinal_loss
            )


        scaler.scale(
            total_batch_loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        total_loss += (
            total_batch_loss.item()
        )


        total_binary_loss += (
            binary_loss.item()
        )


        total_choice_loss += (
            choice_loss.item()
        )


        total_ordinal_loss += (
            ordinal_loss.item()
        )


        num_batches += 1


    divisor = max(
        num_batches,
        1
    )


    return {
        "loss": total_loss / divisor,
        "binary_loss": total_binary_loss / divisor,
        "choice_loss": total_choice_loss / divisor,
        "ordinal_loss": total_ordinal_loss / divisor,
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()


    total_loss = 0.0

    total_binary_loss = 0.0

    total_choice_loss = 0.0

    total_ordinal_loss = 0.0


    total_binary_correct = 0

    total_choice_correct = 0


    total_score_absolute_error = 0.0

    total_score_correct = 0


    total_samples = 0

    total_batches = 0


    for batch in val_loader:

        question_input_ids = batch[
            "question_input_ids"
        ].to(device)


        question_attention_mask = batch[
            "question_attention_mask"
        ].to(device)


        option_input_ids = batch[
            "option_input_ids"
        ].to(device)


        option_attention_mask = batch[
            "option_attention_mask"
        ].to(device)


        choice_labels = batch[
            "choice"
        ].to(device)


        binary_labels = batch[
            "yes"
        ].long().to(device)


        score_labels = batch[
            "score"
        ].to(device)


        # Convert:
        #
        # 0 -> 1
        # 1 -> 2
        # 2 -> 3
        # 3 -> 4
        # 4 -> 5

        real_score_labels = (
            score_labels + 1
        )


        if use_amp:

            autocast_context = torch.amp.autocast(
                "cuda",
                enabled=True
            )

        else:

            autocast_context = torch.amp.autocast(
                "cpu",
                enabled=False
            )


        with autocast_context:

            question_output = model.forward_question(
                question_input_ids,
                question_attention_mask
            )


            option_logits = model.score_options(
                option_input_ids,
                option_attention_mask
            )


            binary_logits = question_output[
                "binary"
            ]


            ordinal_logits = question_output[
                "ordinal"
            ]


            binary_loss = binary_loss_fn(
                binary_logits,
                binary_labels
            )


            choice_loss = choice_loss_fn(
                option_logits,
                choice_labels
            )


            ordinal_loss = ordinal_loss_fn(
                ordinal_logits,
                score_labels
            )


            total_batch_loss = (
                binary_loss
                + choice_loss
                + ORDINAL_WEIGHT * ordinal_loss
            )


        total_loss += (
            total_batch_loss.item()
        )


        total_binary_loss += (
            binary_loss.item()
        )


        total_choice_loss += (
            choice_loss.item()
        )


        total_ordinal_loss += (
            ordinal_loss.item()
        )


        # ----------------------------------------------------
        # Binary prediction
        # ----------------------------------------------------

        binary_predictions = torch.argmax(
            binary_logits,
            dim=1
        )


        total_binary_correct += (
            binary_predictions
            == binary_labels
        ).sum().item()


        # ----------------------------------------------------
        # Choice prediction
        # ----------------------------------------------------

        choice_predictions = torch.argmax(
            option_logits,
            dim=1
        )


        total_choice_correct += (
            choice_predictions
            == choice_labels
        ).sum().item()


        # ----------------------------------------------------
        # Score prediction
        # ----------------------------------------------------

        expected_scores = ordinal_expected_score(
            ordinal_logits
        )


        score_error = torch.abs(
            expected_scores
            - real_score_labels.float()
        )


        total_score_absolute_error += (
            score_error.sum().item()
        )


        rounded_scores = torch.round(
            expected_scores
        ).clamp(
            min=1,
            max=5
        ).long()


        total_score_correct += (
            rounded_scores
            == real_score_labels
        ).sum().item()


        batch_size = (
            binary_labels.size(0)
        )


        total_samples += batch_size

        total_batches += 1


    divisor = max(
        total_batches,
        1
    )


    return {
        "loss": total_loss / divisor,

        "binary_loss": (
            total_binary_loss
            / divisor
        ),

        "choice_loss": (
            total_choice_loss
            / divisor
        ),

        "ordinal_loss": (
            total_ordinal_loss
            / divisor
        ),

        "binary_accuracy": (
            total_binary_correct
            / max(total_samples, 1)
        ),

        "choice_accuracy": (
            total_choice_correct
            / max(total_samples, 1)
        ),

        "score_mae": (
            total_score_absolute_error
            / max(total_samples, 1)
        ),

        "score_accuracy": (
            total_score_correct
            / max(total_samples, 1)
        ),
    }


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float(
    "inf"
)

epochs_without_improvement = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 70)


    train_metrics = train_one_epoch()

    val_metrics = validate()


    # --------------------------------------------------------
    # Training output
    # --------------------------------------------------------

    print(
        f"Train Loss       : "
        f"{train_metrics['loss']:.4f}"
    )


    print(
        f"  Binary Loss     : "
        f"{train_metrics['binary_loss']:.4f}"
    )


    print(
        f"  Choice Loss     : "
        f"{train_metrics['choice_loss']:.4f}"
    )


    print(
        f"  Ordinal Loss    : "
        f"{train_metrics['ordinal_loss']:.4f}"
    )


    print(
        f"Validation Loss   : "
        f"{val_metrics['loss']:.4f}"
    )


    print(
        f"Binary Accuracy   : "
        f"{val_metrics['binary_accuracy']:.4f}"
    )


    print(
        f"Choice Accuracy   : "
        f"{val_metrics['choice_accuracy']:.4f}"
    )


    print(
        f"Score MAE         : "
        f"{val_metrics['score_mae']:.4f}"
    )


    print(
        f"Score Accuracy    : "
        f"{val_metrics['score_accuracy']:.4f}"
    )


    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if (
        val_metrics["loss"]
        < best_val_loss
    ):

        best_val_loss = (
            val_metrics["loss"]
        )

        epochs_without_improvement = 0


        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_loss":
                    best_val_loss,

                "config": {
                    "version": "0.4.1",
                    "model_name": MODEL_NAME,
                    "max_length": MAX_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                }
            },
            BEST_CHECKPOINT
        )


        print()
        print(
            "✓ Best checkpoint saved:"
        )

        print(
            BEST_CHECKPOINT
        )


    else:

        epochs_without_improvement += 1


        print()
        print(
            "No improvement."
        )


        print(
            f"Patience: "
            f"{epochs_without_improvement}"
            f"/{PATIENCE}"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("          DRISTI v0.4.1 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best checkpoint: "
    f"{BEST_CHECKPOINT}"
)

print("=" * 70)