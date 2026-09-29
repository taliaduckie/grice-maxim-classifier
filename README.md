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
*violating* (unintentional failure) as a secondary label where evidence permits. Hopefully, anyways. Issues gotten into below!

## Theoretical grounding

- Grice (1975), "Logic and Conversation"
- Horn (1972, 1984), neo-Gricean Q/R principles
- Levinson (2000), presumptive meanings
- Cutting (2002), flouting vs. violating

## Model

Fine-tuned 'roberta-base', all layers unfrozen. Training: 10 epochs, lr=1e-5,
batch size 8, class-weighted loss, ~15 minutes on CPU. The best epoch is
selected on a dev split and snapshotted in memory ('training_utils.KeepBestState')
because the Trainer's own checkpoint reload silently drops LayerNorm weights in
this transformers version. 'train.py' refuses to save a model that does not
reproduce the score it was selected for.

### Results on the frozen test sets

Evaluated with 'src/evaluate.py' against the splits frozen in
'data/test/manifest.json'. Raw output: 'results/eval_roberta-grice.json'.
95% CIs from 500 bootstrap resamples.

| Test set | n | Macro F1 | 95% CI | Accuracy |
|---|---|---|---|---|
| 'test_synthetic' (hand-written, balanced) | 73 | **0.88** | [0.79, 0.94] | 0.88 |
| 'test_natural' (Reddit, 4 threads) | 50 | 0.32 | [0.16, 0.45] items / [0.09, 0.38] threads | 0.26 |

Two reference points from 'results/foundation_report.md':

- On 'test_natural', **no model beats chance.** Drawing labels from the class
  priors scores 0.21 macro F1; the best of a 4-config x 3-seed RoBERTa ablation
  grid scores 0.26, and TF-IDF + logistic regression 0.25. Every CI overlaps
  every other and overlaps chance.
- On 'test_synthetic', a model that cannot read words (13 features: lengths,
  punctuation counts, word overlap) scores **0.60** macro F1. The synthetic
  corpus is largely separable by shape.

The saved checkpoint predicts Cooperative for 42% of 'test_natural', whose true
Cooperative rate is 0%. That is the register-to-Cooperative shortcut described
in the report: the natural half of the corpus is 59% Cooperative and written in
a different register from the synthetic half, so "looks like Reddit" predicts
"Cooperative" at three times base rate.

**Caveats on the natural number.** 'test_natural' is comment-to-reply, not
question-to-answer (4% of its contexts contain a question, against 88% of the
synthetic set), it comes from only 4 threads, and its gold labels are 94%
flouting with zero Cooperative, which is not a plausible description of a
natural sample. A rebuilt question-to-answer set of 187 pairs across 50 threads
('data/derived/natural_qa_pairs.csv') is awaiting independent annotation under
'docs/annotation_guidelines.md'. Until that lands, treat the natural-domain
direction as established and its magnitude as not. The checkpoint's training
data path is also not recorded in the artefact; the frozen test rows are
excluded from 'data/derived/corpus_train.csv' by test, but a retrain from that
file is the clean way to reproduce the table.

### Adversarial set

On the 40-pair set designed to strip surface cues
('data/annotated/ambiguous_exchanges.csv'), macro F1 = 0.26 (accuracy 42.5%).
Claude (Sonnet 4) on the same items reaches macro F1 = 0.48 (accuracy 55%).
Agreement between the two is 27.5%; RoBERTa defaults to Manner and Claude to
Quantity on the same vague replies, which is where the taxonomy is weakest.
Those runs used a checkpoint saved before the LayerNorm fix, so they are due
to be re-run.

### Withdrawn numbers

Earlier versions of this README reported macro F1 = 0.91 on the 367-pair
synthetic corpus and ~0.77 on the 1,197-pair extended corpus. 'paper.pdf'
reports the 0.91 figure. Both are withdrawn:

- The eval split was also the split used to pick the best epoch, so the score
  was selection-optimal over ten epochs rather than held-out.
- 48 of the 50 human-annotated Reddit "gold" items were in the training corpus.
- The saved models mixed best-epoch weights with final-epoch LayerNorms (see
  above), so the artefact that was evaluated was not the one that was scored.

The 0.88 on 'test_synthetic' is the replacement for 0.91. There is no honest
replacement for 0.77 yet; the closest is the table above.

Fallback: 'facebook/bart-large-mnli' zero-shot if no fine-tuned model exists.
Used to bootstrap annotation; it scores 0.13 macro F1 on the synthetic fold,
below chance for a five-class task.

## Corpus

**'data/annotated/corpus.csv'** — the full 1,197-pair extended corpus.
'data/annotated/corpus_367.csv' is the first 367 rows, the version the paper
analysed. **Train from 'data/derived/corpus_train.csv'** (976 rows), which is
the corpus with the frozen test rows removed; 'tests/test_test_set.py' fails
if a training row and a test row ever share an id or a text.
'data/derived/corpus_provenance.csv' tags every row with source and subreddit.

| Subset | Pairs | Source |
|---|---|---|
| Synthetic (paper) | 367 | Hand-written by author, bootstrapped via BART-MNLI |
| Reddit additions | ~830 | r/AmItheAsshole, r/explainlikeimfive, r/cscareerquestions, r/relationships, r/relationship_advice, r/ExperiencedDevs, r/askscience, r/MaliciousCompliance, r/tifu, r/talesfromtechsupport |

The synthetic 367 was built through five rounds of bootstrapping plus several
targeted batches (sarcasm, opting-out, balancing). The Reddit additions were
scraped via 'src/scrape_reddit.py', pre-labeled by the fine-tuned model, and
hand-corrected.

## Setup

'''bash
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
python src/train.py --data data/derived/corpus_train.csv

# Evaluate a checkpoint against the frozen test sets
python src/evaluate.py --model models/roberta-grice

# K-fold cross-validation
python src/kfold_eval.py --data data/derived/corpus_train.csv --folds 5

# Coherence scoring for annotation QA
python src/score_corpus.py

# Compare RoBERTa vs Claude on the adversarial set
export ANTHROPIC_API_KEY=sk-ant-...
python src/compare_classifiers.py

# Gradio web demo
python src/app.py

# FastAPI backend
python src/api.py
'''

## Project structure

'''
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
│   ├── llm_annotate.py       # Claude as one annotator over a sheet
│   ├── kfold_eval.py         # stratified k-fold cross-validation
│   ├── score_corpus.py       # coherence scoring for annotation QA
│   ├── compare_classifiers.py # RoBERTa vs Claude API comparison
│   ├── app.py                # Gradio web demo
│   └── api.py                # FastAPI backend
├── data/
│   ├── raw/                  # unannotated and pre-labeled CSVs
│   ├── annotated/            # gold-labeled CSV corpus
│   ├── derived/              # generated splits + provenance (rebuilt, not hand-edited)
│   ├── feedback/             # corrections logged from the demo
│   └── test/                 # frozen held-out test set + manifest
├── docs/                     # annotation guidelines
├── results/                  # eval outputs, baselines, reports
├── models/                   # saved checkpoints (gitignored)
└── tests/
    ├── test_labels.py
    ├── test_corpus.py
    ├── test_predict.py
    └── ...                   # one per module, 15 in all
'''

## TODO

- **Violation type prediction** — right now 'violation_type' is a heuristic. Essentially cooperative = none, everything else = unknown. The corpus has enough flouting/violating/none examples to train a second head or a separate model, but the distinction is often not visible in surface form (especially for Relation, where flouting and violating cluster identically by coherence score).
- **Annotate the rebuilt natural test set** — 'data/derived/natural_qa_pairs.csv', two annotators minimum, then 'src/agreement.py' and 'src/finalize_test_set.py'. Everything natural-domain above is provisional until this lands.
- **Run Claude as a provisional third annotator** — 'src/llm_annotate.py' labels a sheet under the same guidelines a human sees and writes to 'data/test/annotations/<model>.csv'. Blocked on a valid 'ANTHROPIC_API_KEY' (the current one returns 401). Once it works: run it on 'data/derived/natural_qa_pairs.csv' and, with '--out' under 'results/', on 'results/gold_recheck_sheet.csv', then compare against the human passes with 'src/agreement.py'. Its labels are a check on the humans, not a substitute for them.
- **Re-run the adversarial comparison** — 'src/compare_classifiers.py' still pins a 2025 Sonnet model ID and its numbers above came from a pre-LayerNorm-fix checkpoint. Update the model, re-run against the current checkpoint, and replace the 0.26 / 0.48 figures. Same API-key blocker.
- **Try roberta-large** — twice the parameters, so probably a few F1 points for free! Worth poking.
- **Adversarial set expansion** — 40 items is small. Larger adversarial coverage would let macro F1 carry a lot more statistical weight.

## License

This repository — the code, the annotated corpus, and the documentation — is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). See [LICENSE](LICENSE) for the full text.

You are free to share and adapt this material for any purpose, including commercially, as long as you give appropriate credit, link to the license, and indicate if changes were made.

Reddit-sourced rows in `data/raw/` are excerpts of user-generated content used under Reddit's terms; the CC BY license covers the annotations, curation, and hand-authored examples, not the underlying Reddit text.
