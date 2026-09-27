import torch
import torch.nn as nn
from transformers import AutoModel


class DristiModelV04(nn.Module):

    def __init__(
        self,
        model_name="microsoft/deberta-v3-small",
        num_scores=5
    ):
        super().__init__()

        # =================================================
        # SHARED ENCODER
        # =================================================

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        # =================================================
        # BINARY DECISION HEAD
        #
        # Output:
        # class 0 = NO
        # class 1 = YES
        # =================================================

        self.binary_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                256,
                2
            )
        )

        # =================================================
        # DYNAMIC OPTION SCORER
        #
        # Input:
        # question + option
        #
        # Output:
        # one score for each option
        # =================================================

        self.option_scorer = nn.Sequential(

            nn.Linear(
                hidden_size,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                256,
                1
            )
        )

        # =================================================
        # ORDINAL SCORE HEAD
        #
        # For scores 1-5 we predict four thresholds:
        #
        # P(score > 1)
        # P(score > 2)
        # P(score > 3)
        # P(score > 4)
        #
        # This respects the ordering of the scores.
        # =================================================

        self.ordinal_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                256,
                num_scores - 1
            )
        )

    # =====================================================
    # ENCODER
    # =====================================================

    def encode(
        self,
        input_ids,
        attention_mask
    ):

        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        state = (
            output.last_hidden_state[:, 0]
        )

        return state

    # =====================================================
    # QUESTION OUTPUTS
    # =====================================================

    def forward_question(
        self,
        input_ids,
        attention_mask
    ):

        state = self.encode(
            input_ids,
            attention_mask
        )

        binary_logits = (
            self.binary_head(state)
        )

        ordinal_logits = (
            self.ordinal_head(state)
        )

        return {
            "state": state,
            "binary": binary_logits,
            "ordinal": ordinal_logits
        }

    # =====================================================
    # DYNAMIC OPTION SCORING
    #
    # Input shape:
    #
    # [batch, options, sequence]
    #
    # Example:
    #
    # [2, 4, 128]
    #
    # = 2 questions
    # = 4 options each
    # = 128 tokens
    # =====================================================

    def score_options(
        self,
        option_input_ids,
        option_attention_mask
    ):

        batch_size = (
            option_input_ids.shape[0]
        )

        num_options = (
            option_input_ids.shape[1]
        )

        sequence_length = (
            option_input_ids.shape[2]
        )

        # -------------------------------------------------
        # Flatten batch + options
        # -------------------------------------------------

        flat_input_ids = (
            option_input_ids.reshape(
                batch_size * num_options,
                sequence_length
            )
        )

        flat_attention_mask = (
            option_attention_mask.reshape(
                batch_size * num_options,
                sequence_length
            )
        )

        # -------------------------------------------------
        # Encode all question-option pairs
        # -------------------------------------------------

        option_state = self.encode(
            flat_input_ids,
            flat_attention_mask
        )

        # -------------------------------------------------
        # Score every option
        # -------------------------------------------------

        logits = (
            self.option_scorer(option_state)
            .squeeze(-1)
        )

        # -------------------------------------------------
        # Restore:
        #
        # [batch, options]
        # -------------------------------------------------

        logits = logits.reshape(
            batch_size,
            num_options
        )

        return logits