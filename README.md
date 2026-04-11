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

Fine-tuned: `roberta-base` on 367 hand-annotated examples. Macro F1 = **0.86**
on a stratified 80/20 eval split. Training: 10 epochs, lr=2e-5, batch size 4,
~9 minutes on CPU.

Per-class accuracy on the full corpus:

| Maxim | Accuracy |
|---|---|
| Quantity | 99% |
| Relation | 99% |
| Manner | 97% |
| Cooperative | 96% |
| Quality | 93% |

Sarcasm detection was the hardest problem. 20 targeted workplace sarcasm
examples got Quality flouting from ~65% confidence to 99%+.

Fallback: `facebook/bart-large-mnli` zero-shot baseline if no fine-tuned model
exists. Works by passing natural-language hypothesis descriptions to the NLI
model and scoring entailment. Good enough to bootstrap annotation but not
much else.

## Corpus

367 annotated utterance-context pairs in `data/annotated/corpus.csv`.
Distribution: 86 Quality, 72 Quantity, 70 Relation, 70 Cooperative, 69 Manner.
150 violating, 147 flouting, 70 none.

Bootstrapped via `src/bootstrap.py`, which runs zero-shot predictions on
seed pairs and outputs a CSV for human correction. The model's guesses
are wrong often enough to keep you honest and right often enough to be
faster than annotating from scratch. Five rounds of bootstrap + annotate
plus targeted rebalancing got us from 8 examples to 367.

## Setup

```bash
pip install -r requirements.txt

# Single utterance inference
python src/predict.py --text "The weather is nice today." \
                      --context "Why were you late to the meeting?"

# Batch mode — run on a CSV, compare against gold labels
python src/predict.py --batch data/annotated/corpus.csv
python src/predict.py --batch data/annotated/corpus.csv --output results.csv

# Bootstrap more annotations
python src/bootstrap.py

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
│   ├── predict.py      # inference CLI
│   └── bootstrap.py    # pre-label seed pairs for annotation
├── data/
│   ├── raw/            # unannotated utterance pairs
│   └── annotated/      # gold-labeled CSV corpus
├── models/             # saved checkpoints
└── tests/
    └── test_labels.py
```

## TODO

- **K-fold cross-validation** — the single 80/20 stratified split means F1 numbers depend on which 50 examples land in eval. K-fold would give more stable estimates and catch classes that happen to get lucky or unlucky in a given split.
- **Violation type prediction** — right now `violation_type` is a heuristic: cooperative = none, everything else = flouting. The corpus has 147 flouting, 150 violating, 70 none — enough to train a second classification head or a separate model. The flouting/violating distinction is the interesting part of Gricean pragmatics and we're currently just guessing.
