# Foundation report — held-out evaluation, provenance, and baselines

Covers checklist items 1, 2 (tooling), 4 (provenance), 7, and 10.
Generated from `src/build_test_set.py`, `src/provenance.py`, `src/baselines.py`.
Raw numbers: `results/baselines.json`. Test set frozen at `data/test/manifest.json`.

---

## 0. The finding, up front

**On genuinely held-out natural data, no baseline beats chance.** The best macro
F1 on `test_natural` is 0.254, 95% CI [0.13, 0.37]; random guessing from the
class priors scores 0.212. TF-IDF + logistic regression is not statistically
distinguishable from always guessing the majority class (McNemar p = 0.125).

**On the synthetic corpus, a model that cannot read words scores 0.599** — 88%
of the best lexical model's 0.681, and the gap is not significant (p = 0.345).
The 13 features it uses are lengths, punctuation counts, and word overlap. The
synthetic corpus is separable by shape alone.

Those two results together are the paper. Not "we built a classifier."

---

## 1. Contamination in the previous numbers

`gold_annotated.csv` — the 50 human-annotated Reddit pairs — was **48/50 already
present in `corpus.csv`**, the training corpus. Any score previously computed on
those items was a training score.

Two of its 50 rows were also structurally broken: a context field containing a
comma had been written without quoting, splitting one record across 10 and
another across 11 fields. `pandas.read_csv` refuses the whole file; the naive
`csv` reader silently shifts every column, so those rows carried
`gold_maxim = "0.979"` and `subreddit = "flout"`. Both are now repaired
(`src/provenance.py:repair_row`) and all 50 rows are usable.

## 2. What now exists

| File | Rows | What it is |
|---|---|---|
| `data/test/test_natural.csv` | 50 | Human-annotated Reddit pairs. Frozen. |
| `data/test/test_synthetic.csv` | 73 | Stratified 20% of the 367 hand-written pairs. Frozen. |
| `data/test/test_natural_pending.csv` | 100 | Natural pairs, **labels stripped**, awaiting independent annotation. |
| `data/annotated/corpus_train.csv` | 976 | Everything else: 294 synthetic + 682 natural. |
| `data/annotated/corpus_provenance.csv` | 1197 | Full corpus with `source`, `subreddit`, `row_id`. |
| `data/test/manifest.json` | — | Row-id hashes per split, so drift is detectable. |

221 rows were removed from training to build this. `tests/test_test_set.py`
fails if a training row and a test row ever share an id *or* a text.

Provenance recovery worked for 634 of 830 natural rows (subreddit joined back
from `data/raw/*.csv`); the remaining 196 are marked natural but unattributed.

## 3. The confound, quantified

Provenance tagging makes the register→Cooperative shortcut arithmetic rather
than suspicion:

| Maxim | Synthetic (n=367) | Natural (n=830) |
|---|---|---|
| Cooperative | 70 (19%) | 490 (**59%**) |
| Quality | 86 | 142 (17%) |
| Quantity | 70 | 92 (11%) |
| Manner | 71 | 63 (8%) |
| Relation | 70 | 43 (5%) |

The synthetic half is balanced by construction; the natural half is 59%
Cooperative and written in a completely different register. "Looks like Reddit"
therefore predicts "Cooperative" at almost three times base rate, and that is a
free lunch any model will take.

It does. Percentage of `test_natural` predicted Cooperative, by training regime:

| Trained on | % predicted Cooperative | True rate |
|---|---|---|
| synthetic only | 0–18% | 0% |
| natural only | 58–82% | 0% |
| both | 42–80% | 0% |

Adding natural training data makes natural-domain predictions *worse*, because
what it mostly teaches is the prior.

## 4. Baselines

Macro F1 over classes present in the gold split. 95% CI from 1,000 bootstrap
resamples of the test set; ± is the SD over 10 bootstrap resamples of the
training set. Hyperparameters fixed in source, never tuned against these sets.

**Trained on synthetic only (294 rows)**

| Model | test_natural | test_synthetic |
|---|---|---|
| majority | 0.090 [0.05, 0.13] | 0.076 [0.05, 0.10] |
| stratified (chance) | 0.212 [0.10, 0.32] | 0.230 [0.14, 0.32] |
| surface (no words) | 0.173 [0.09, 0.25] | **0.599** [0.49, 0.70] |
| tfidf-utt | **0.254** [0.13, 0.37] | 0.639 [0.52, 0.74] |
| tfidf-ctx | 0.213 [0.12, 0.29] | 0.664 [0.54, 0.77] |
| tfidf-char | 0.234 [0.12, 0.35] | 0.681 [0.56, 0.79] |

**Trained on natural only (682 rows)**

| Model | test_natural | test_synthetic |
|---|---|---|
| majority | 0.000 | 0.064 |
| stratified | 0.139 [0.03, 0.25] | 0.134 [0.06, 0.21] |
| surface | **0.210** [0.10, 0.33] | 0.328 [0.23, 0.41] |
| tfidf-utt | 0.135 [0.02, 0.24] | 0.399 [0.28, 0.50] |
| tfidf-ctx | 0.124 [0.03, 0.23] | 0.395 [0.26, 0.50] |
| tfidf-char | 0.095 [0.00, 0.19] | 0.454 [0.33, 0.56] |

**Trained on both (976 rows)**

| Model | test_natural | test_synthetic |
|---|---|---|
| majority | 0.000 | 0.064 |
| stratified | 0.082 [0.00, 0.17] | 0.232 [0.13, 0.32] |
| surface | 0.088 [0.02, 0.16] | 0.494 [0.40, 0.57] |
| tfidf-utt | **0.148** [0.03, 0.25] | **0.760** [0.65, 0.85] |
| tfidf-ctx | 0.119 [0.03, 0.21] | 0.600 [0.49, 0.70] |
| tfidf-char | 0.148 [0.03, 0.26] | 0.683 [0.57, 0.77] |

Every confidence interval on `test_natural` overlaps chance. Every confidence
interval on `test_natural` also overlaps every other model's. With 50 items
there is no resolving power to distinguish these — which is itself a result
about the evaluation, and the reason `test_natural_pending.csv` exists.

## 5. Significance tests (exact McNemar, paired)

| Comparison | test_natural | test_synthetic |
|---|---|---|
| tfidf-ctx vs majority | p = 0.125 — **ns** | p < 0.0001 — sig |
| tfidf-ctx vs tfidf-utt (does context help?) | p = 1.000 — ns | p = 0.093 — ns |
| tfidf-ctx vs surface (do words help?) | p = 1.000 — ns | p = 0.359 — **ns** |
| tfidf-char vs surface, synthetic-trained | p = 1.000 — ns | p = 0.345 — **ns** |
| synthetic-trained vs natural-trained | p = 0.012 — sig | p = 0.002 — sig |

Three things fall out:

1. **Lexical content does not significantly beat surface statistics** on either
   test set. On the synthetic corpus both score well; the model reading actual
   words is not doing meaningfully better than one counting characters.
2. **Context does not help** (item 3). Adding the prior turn to the TF-IDF
   features changes nothing on natural data and slightly *hurts* on synthetic.
   Worth re-testing at RoBERTa scale before concluding anything general, since a
   bag of n-grams cannot represent the relation between the turns — but the
   burden of proof now sits with the context-aware model.
3. **Training domain matters and transfers badly in both directions.**

## 6. Limitation that has to be stated with every natural number

`test_natural` contains **zero Cooperative examples** — all 50 gold items are
violations. Consequences:

- It measures "given that a violation occurred, which one?", not detection.
- Any Cooperative prediction is automatically wrong, so the register shortcut is
  maximally penalised. The natural-domain numbers are a *lower* bound.
- Macro F1 is over 4 classes on natural and 5 on synthetic — not comparable
  as absolute values, only as directions.

`test_natural_pending.csv` fixes this: 100 rows sampled across the natural
distribution, 62 of which currently carry a Cooperative label (hidden from the
annotator). Once annotated, the natural test set becomes ~150 rows with real
Cooperative coverage and the numbers above should be recomputed.

## 7. What is ready and waiting on you

- `docs/annotation_guidelines.md` — v1.0. Decision procedure with an explicit
  ordering (Relation → Quality → Quantity → Manner → Cooperative), a flouting
  test, 12 stipulated conventions for recurring cases, a 10-item calibration
  set, and the anti-anchoring rules.
- `src/agreement.py` — percent agreement, Cohen's κ, Fleiss' κ, Krippendorff's α
  (validated against the reference implementation on 7 cases including missing
  data), per-label agreement, disagreement confusion, and an auto-generated
  adjudication sheet.

The blocking step is human: two or three people annotating
`test_natural_pending.csv` independently, then `python3 src/agreement.py
data/test/annotations/*.csv`. Until α is known, every model number in this
report has an unknown ceiling.

## 8. Not yet done

- Item 8's RoBERTa ablation grid — the cheap models already show the shape, but
  the transformer numbers are what the paper claims.
- Items 5, 6, 9, 11 — hard-phenomenon subsets, minimal pairs, task
  decomposition, 100-failure error analysis.
- `README.md` still reports macro F1 0.91 and 0.77 from splits that included the
  gold items and had no provenance separation. Those numbers need re-deriving
  against `corpus_train.csv` before they are quoted anywhere.
