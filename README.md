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

## Setup

```bash
pip install -r requirements.txt

# Zero-shot inference (no training needed)
python src/predict.py --text "The weather is nice today." \
                      --context "Why were you late to the meeting?"

# Fine-tune on your annotated data
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
│   └── predict.py      # inference CLI
├── data/
│   ├── raw/            # unannotated utterance pairs
│   └── annotated/      # gold-labeled CSV corpus
├── models/             # saved checkpoints
└── tests/
    └── test_labels.py
```
