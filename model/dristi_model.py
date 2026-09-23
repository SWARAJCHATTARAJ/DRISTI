import torch.nn as nn
from transformers import AutoModel


class DristiModel(nn.Module):

    def __init__(
        self,
        model_name="distilbert/distilbert-base-uncased",
        num_scores=5
    ):
        super().__init__()

        # Shared language encoder
        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.encoder.config.hidden_size

        # 1. Binary decision head
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 2. Dynamic option scorer
        self.option_scorer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 3. Score head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_scores)
        )

    def encode(self, input_ids, attention_mask):

        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # First token representation
        state = output.last_hidden_state[:, 0]

        return state

    def forward_question(
        self,
        input_ids,
        attention_mask
    ):

        state = self.encode(
            input_ids,
            attention_mask
        )

        binary_logits = self.binary_head(
            state
        ).squeeze(-1)

        score_logits = self.score_head(
            state
        )

        return {
            "state": state,
            "binary": binary_logits,
            "score": score_logits
        }

    def score_options(
        self,
        option_input_ids,
        option_attention_mask
    ):

        option_state = self.encode(
            option_input_ids,
            option_attention_mask
        )

        option_logits = self.option_scorer(
            option_state
        ).squeeze(-1)

        return option_logits