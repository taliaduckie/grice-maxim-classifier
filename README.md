# Grice Maxim Classifier

A transformer-based classifier for detecting Gricean maxim violations
in natural language utterances.

Classifies across five categories:
- **Quantity** — too much or too little information
- **Quality** — apparent falsehood or unsupported assertion
- **Relation** — apparent irrelevance to the discourse context
- **Manner** — obscurity, ambiguity, prolixity, or disorder
- **Cooperative** — fully cooperative (no apparent violation)

Also distinguishes *flouting* (deliberate, to generate implicature) from
*violating* (unintentional failure) as a secondary label where evidence permits.

## Theoretical grounding

- Grice (1975), "Logic and Conversation"
- Horn (1984), neo-Gricean Q/R principles
- Levinson (2000), presumptive meanings
- Cutting (2002), flouting vs. violating

## Model

Zero-shot baseline: `facebook/bart-large-mnli`
Fine-tuned target: `roberta-base` on a manually annotated corpus

The zero-shot baseline works by passing natural-language hypothesis
descriptions of each maxim to the NLI model and scoring entailment.
It gets Quantity reliably, Relation sometimes, and Manner almost never.
Quality flouting (irony, hyperbole) is hard for reasons that are hard.

## Corpus

40 annotated utterance-context pairs in `data/annotated/corpus.csv`.
Distribution: 9 Cooperative, 7 Quantity, 8 Quality, 7 Relation, 6 Manner.
Heavily skewed toward flouting (26) over violating (5) — the interesting
cases are the deliberate ones.

Bootstrapped via `src/bootstrap.py`, which runs zero-shot predictions on
seed pairs and outputs a CSV for human correction. The model's guesses
are wrong often enough to keep you honest and right often enough to be
faster than annotating from scratch.

## Setup

```bash
pip install -r requirements.txt

# Zero-shot inference (no training needed)
python src/predict.py --text "The weather is nice today." \
                      --context "Why were you late to the meeting?"

# Bootstrap more annotations
python src/bootstrap.py

# Fine-tune on your annotated data (once you have enough of it)
python src/train.py --data data/annotated/corpus.csv
```

## Project structure

```
grice-maxim-classifier/
├── src/
│   ├── labels.py       # maxim definitions and label schema
│   ├── zero_shot.py    # zero-shot baseline (BART-MNLI)
│   ├── dataset.py      # dataset loading and tokenization
│   ├── train.py        # fine-tuning loop (RoBERTa)
│   ├── predict.py      # inference CLI
│   └── bootstrap.py    # pre-label seed pairs for annotation
├── data/
│   ├── raw/            # unannotated utterance pairs
│   └── annotated/      # gold-labeled CSV corpus
├── models/             # saved checkpoints
└── tests/
    └── test_labels.py
```
