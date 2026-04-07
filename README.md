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

Fine-tuned: `roberta-base` on 249 hand-annotated examples. Macro F1 = **0.81**
on a stratified 80/20 eval split. Training: 10 epochs, lr=2e-5, batch size 4.

Sarcasm detection was the hardest problem. 20 targeted workplace sarcasm
examples got Quality flouting from ~65% confidence to 99%+.

Fallback: `facebook/bart-large-mnli` zero-shot baseline if no fine-tuned model
exists. Works by passing natural-language hypothesis descriptions to the NLI
model and scoring entailment. Good enough to bootstrap annotation but not
much else.

## Corpus

249 annotated utterance-context pairs in `data/annotated/corpus.csv`.
Distribution: 69 Quality, 49 Quantity, 46 Manner, 45 Relation, 40 Cooperative.
147 flouting, 62 violating, 40 none.

Bootstrapped via `src/bootstrap.py`, which runs zero-shot predictions on
seed pairs and outputs a CSV for human correction. The model's guesses
are wrong often enough to keep you honest and right often enough to be
faster than annotating from scratch. Five rounds of bootstrap + annotate
got us from 8 examples to 249.

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
