"""
dataset.py
Load and tokenize the annotated corpus for fine-tuning.

Expected CSV format:
    utterance, context, maxim, violation_type
"""

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from labels import MAXIMS

LABEL2ID = {m: i for i, m in enumerate(MAXIMS)}
ID2LABEL = {i: m for m, i in LABEL2ID.items()}
MODEL_NAME = "roberta-base"


class GriceDataset(Dataset):
    def __init__(self, csv_path: str, max_length: int = 256):
        df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.encodings = self.tokenizer(
            list(df["utterance"]),
            list(df.get("context", [""] * len(df))),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [LABEL2ID[m] for m in df["maxim"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
