import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


MODEL_NAME = "distilbert/distilbert-base-uncased"


class DristiDataset(Dataset):

    def __init__(
        self,
        file_path,
        max_length=128
    ):

        # Load JSON dataset
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.data = json.load(file)

        # Load tokenizer
        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        self.max_length = max_length

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        # -----------------------------------------
        # Encode the question
        # -----------------------------------------

        question = self.tokenizer(
            item["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # -----------------------------------------
        # Encode question + every option
        # -----------------------------------------

        option_texts = []

        for option in item["options"]:

            combined_text = (
                item["question"]
                + " [SEP] "
                + option
            )

            option_texts.append(
                combined_text
            )

        options = self.tokenizer(
            option_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {

            # Question
            "question_input_ids":
                question["input_ids"].squeeze(0),

            "question_attention_mask":
                question["attention_mask"].squeeze(0),

            # Options
            "option_input_ids":
                options["input_ids"],

            "option_attention_mask":
                options["attention_mask"],

            # Correct option index
            "choice":
                torch.tensor(
                    item["choice"],
                    dtype=torch.long
                ),

            # Yes / No label
            "yes":
                torch.tensor(
                    item["yes"],
                    dtype=torch.float32
                ),

            # Convert score 1-5 → 0-4
            "score":
                torch.tensor(
                    item["score"] - 1,
                    dtype=torch.long
                )
        }