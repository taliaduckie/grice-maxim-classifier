"""
dataset.py

Data loading and tokenization for fine-tuning.

Expected CSV format (see data/annotated/corpus.csv for examples):
    utterance, context, maxim, violation_type
    "The weather is nice.", "Why were you late?", "Relation", "flouting"

violation_type is optional and currently unused during training.
I'm only training on the five-way maxim classification for now.
The violation_type label (flouting vs. violating) is a separate
problem that probably needs more data and a different architecture.

The thing I keep bumping into with this dataset is that maxim violations
are often context-dependent in ways that make annotation really frustrating &
hard. Is "The report is fine" in response to "Is the report ready?"
a Manner violation (what does 'fine' even mean here) or a Quantity
violation (that's less information than I asked for) or a cooperative
response with a pragmatic implicature of mild reluctance? ACK.

My cuurent answer: it depends on what the speaker meant, which you can't
always recover from text alone.  That's a limitation of the dataset but in a
way it's also kind of the point of the whole project i guess.
"""

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from labels import MAXIMS

LABEL2ID = {m: i for i, m in enumerate(MAXIMS)}
ID2LABEL  = {i: m for m, i in LABEL2ID.items()}
MODEL_NAME = "roberta-base"  # Change this if you want to experiment with other models.
                              # roberta-large would probably be better; it's also
                              # twice as slow and twice as expensive to fine-tune.
                              # Your call.


class GriceDataset(Dataset):
    """
    Torch Dataset wrapping the annotated CSV corpus.

    Tokenizes utterance+context pairs and maps maxim labels to integers.
    Context is passed as the second sequence to the tokenizer (segment B),
    which is how RoBERTa expects paired inputs. This matches the structure
    of NLI tasks, which is intentional — we're essentially asking the model
    to learn "does this context-utterance pair exhibit violation X?"
    """

    def __init__(self, csv_path: str, max_length: int = 256):
        """
        Args:
            csv_path:   path to the annotated CSV
            max_length: max token length for truncation.
                        256 is generous for most utterance pairs.
                        Most conversational turns are well under 50 tokens.
                        But the Manner examples tend to run long, which is
                        sort of appropriate given what Manner is.
        """
        df = pd.read_csv(csv_path)

        assert "utterance" in df.columns and "maxim" in df.columns, (
            "CSV must have at least 'utterance' and 'maxim' columns. "
            "See data/annotated/corpus.csv for the expected format."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # If the CSV has a 'context' column, use it.
        # If not, pass empty strings — the model handles this fine.
        contexts = list(df["context"]) if "context" in df.columns else [""] * len(df)

        self.encodings = self.tokenizer(
            list(df["utterance"]),
            contexts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        # Check that all maxim labels are valid before failing silently at training time.
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
