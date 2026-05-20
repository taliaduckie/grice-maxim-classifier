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
- Horn (1972, 1984), neo-Gricean Q/R principles
- Levinson (2000), presumptive meanings
- Cutting (2002), flouting vs. violating

## Model

Fine-tuned `roberta-base`. Training: 10 epochs, lr=1e-5, batch size 8, class
weights, ~15 minutes on CPU.

### Paper version (367-pair synthetic corpus)

The numbers in `paper.pdf` are reported on this version of the corpus.

Macro F1 = **0.91** on a stratified 80/20 split (74 eval, 293 train).

Per-class F1 (held-out fold):

| Maxim | F1 |
|---|---|
| Quantity | 0.97 |
| Relation | 0.97 |
| Cooperative | 0.88 |
| Manner | 0.88 |
| Quality | 0.85 |

Zero-shot BART-MNLI baseline on the same fold: macro F1 = 0.13 (below chance for
a five-class task).

On the 40-pair adversarial set designed to strip surface cues, performance drops
to macro F1 = 0.26 (accuracy 42.5%), suggesting the held-out F1 substantially
overstates the model's grasp of the maxims. Claude (Sonnet 4) on the same
adversarial set reaches macro F1 = 0.48 (accuracy 55%).

### Extended version (1,197-pair corpus with real Reddit data)

The repo also contains an extended corpus that mixes the 367 synthetic pairs with
~830 real comment-reply pairs scraped from Reddit. Training on the full extended
corpus drops held-out macro F1 to ~0.77, which is the more honest number for
real-world generalization. The drop reflects the synthetic-distribution
inflation of the 0.91 figure: real Reddit conversation is messier, more
ambiguous, and skews heavily Cooperative.

The extended corpus was not used for the paper's main results because the paper
analyzes the 367-pair version. It exists for follow-up work and a fairer test
of real-world performance.

Fallback: `facebook/bart-large-mnli` zero-shot baseline if no fine-tuned model
exists. Used to bootstrap annotation; performs near chance for actual inference.

## Corpus

**`data/annotated/corpus.csv`** — the full 1,197-pair extended corpus. To
reproduce the paper, take the first 367 rows (see `data/annotated/corpus_367.csv`).

| Subset | Pairs | Source |
|---|---|---|
| Synthetic (paper) | 367 | Hand-written by author, bootstrapped via BART-MNLI |
| Reddit additions | ~830 | r/AmItheAsshole, r/explainlikeimfive, r/cscareerquestions, r/relationships, r/relationship_advice, r/ExperiencedDevs, r/askscience, r/MaliciousCompliance, r/tifu, r/talesfromtechsupport |

The synthetic 367 was built through five rounds of bootstrapping plus several
targeted batches (sarcasm, opting-out, balancing). The Reddit additions were
scraped via `src/scrape_reddit.py`, pre-labeled by the fine-tuned model, and
hand-corrected.

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

# Scrape Reddit pairs and pre-label them
python src/scrape_reddit.py --subreddit askreddit --limit 50

# Fine-tune
python src/train.py --data data/annotated/corpus.csv

# K-fold cross-validation
python src/kfold_eval.py --data data/annotated/corpus.csv --folds 5

# Coherence scoring for annotation QA
python src/score_corpus.py

# Compare RoBERTa vs Claude on the adversarial set
export ANTHROPIC_API_KEY=sk-ant-...
python src/compare_classifiers.py

# Gradio web demo
python src/app.py

# FastAPI backend
python src/api.py
```

## Project structure

```
grice-maxim-classifier/
├── src/
│   ├── labels.py             # maxim definitions and label schema
│   ├── zero_shot.py          # zero-shot baseline (BART-MNLI)
│   ├── dataset.py            # dataset loading and tokenization
│   ├── train.py              # fine-tuning loop (RoBERTa)
│   ├── predict.py            # inference CLI
│   ├── bootstrap.py          # pre-label seed pairs for annotation
│   ├── scrape_reddit.py      # scrape comment-reply pairs from Reddit
│   ├── merge_corpus.py       # merge new annotations into corpus
│   ├── kfold_eval.py         # stratified k-fold cross-validation
│   ├── score_corpus.py       # coherence scoring for annotation QA
│   ├── compare_classifiers.py # RoBERTa vs Claude API comparison
│   ├── app.py                # Gradio web demo
│   └── api.py                # FastAPI backend
├── data/
│   ├── raw/                  # unannotated and pre-labeled CSVs
│   └── annotated/            # gold-labeled CSV corpus
├── models/                   # saved checkpoints (gitignored)
└── tests/
    ├── test_labels.py
    ├── test_corpus.py
    └── test_predict.py
```

## TODO

- **Violation type prediction** — right now `violation_type` is a heuristic: cooperative = none, everything else = unknown. The corpus has enough flouting/violating/none examples to train a second head or a separate model, but the distinction is often not visible in surface form (especially for Relation, where flouting and violating cluster identically by coherence score).
- **Held-out test set** — eval is currently part of the training loop. A true out-of-sample set would give a more honest score.
- **Try roberta-large** — twice the parameters, probably a few F1 points for free.
- **Adversarial set expansion** — 40 items is small. Larger adversarial coverage would let macro F1 carry real statistical weight.
