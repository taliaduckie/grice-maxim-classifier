import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from labels import MAXIMS

LABEL2ID = {m: i for i, m in enumerate(MAXIMS)}
ID2LABEL  = {i: m for m, i in LABEL2ID.items()}
MODEL_NAME = "roberta-base" 
                              # roberta-large might be better; it's also twice
                              # as slow and twice as expensive to fine tune
                            


class GriceDataset(Dataset):
    """torch Dataset wrapping the annotated CSV"""

    def __init__(self, csv_path: str, max_length: int = 256):
        df = pd.read_csv(csv_path)

        assert "utterance" in df.columns and "maxim" in df.columns, (
            "CSV needs 'utterance' and 'maxim' columns"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # If the CSV has a 'context' column, use it;
        # if not, pass empty strings
        contexts = list(df["context"]) if "context" in df.columns else [""] * len(df)

        # padding to max_length
        self.encodings = self.tokenizer(
            list(df["utterance"]),
            contexts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        # check all maxim labels are valid
        unknown = set(df["maxim"]) - set(MAXIMS)
        if unknown:
            raise ValueError(
                f"Unknown maxim labels in CSV: {unknown}. "
                f"Valid labels are: {MAXIMS}. "
                f"Check your annotation for typos. ('Cooperative' not 'cooperative'. I know.)"
            )

        self.labels = [LABEL2ID[m] for m in df["maxim"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }
