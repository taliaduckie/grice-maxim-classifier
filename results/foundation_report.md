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

## 8. Two defects in the training harness

Found while building the ablation runner. Both affect numbers already reported.

### 8.1 The saved model is not the model that was evaluated

`train.py` used `load_best_model_at_end=True`. In this transformers version the
checkpoint writer stores LayerNorm parameters under the legacy `gamma`/`beta`
names while the reload path looks for `weight`/`bias`. The mismatch is logged as
"missing keys" and then ignored, so **all 25 LayerNorm layers are silently not
restored**. Measured directly: after the reload the model's LayerNorm weights
were unchanged by the load and differed from the checkpoint's stored values by
up to 0.95.

The restored model therefore carries best-epoch weights everywhere except its
LayerNorms, which retain final-epoch values — a combination that was never
evaluated, and it is what `save_model` then wrote to disk. `from_pretrained`
applies the rename correctly, so a plain save/load round-trip is unaffected;
only the Trainer path is broken.

Consequences:

- Every model in `models/` was saved this way, so `predict.py`, `app.py`,
  `api.py`, and `comparison_results.csv` all ran against a hybrid model rather
  than the one whose score was reported.
- The macro F1 printed *during* training is computed on the in-memory model and
  is correct for that epoch. This is why the bug is quiet: the number is right,
  the artefact is wrong.

Fixed in `src/training_utils.py` (`KeepBestState`), which snapshots the best
epoch's weights in RAM and skips serialisation entirely. Both `train.py` and
`ablations.py` now assert that the restored model reproduces the score it was
selected for, and refuse to save it otherwise. Regression tests in
`tests/test_training_utils.py`.

### 8.2 The reported held-out score selected its own epoch

`train.py` passes the same 20% split as `eval_dataset` — used every epoch for
best-model selection — and then reports that split's macro F1 as the held-out
result. The 0.91 in the README is therefore a selection-optimal number over ten
epochs, not a clean held-out one, independently of the contamination in §1.

The ablation runner carves a separate 15% dev split out of each run's own
training pool for epoch selection, and touches the frozen test sets exactly once
per run, after selection is finished.

## 9. RoBERTa ablation grid (item 8)

4 configs × 3 seeds, `src/ablations.py`, raw output in `results/ablations.json`.
Each run selects its epoch on a 15% dev split carved from its own training pool
and touches the frozen test sets once, afterwards.

| cfg | context | trained on | n | test_natural | test_synthetic | % Coop on natural |
|---|---|---|---|---|---|---|
| A | no | synthetic | 249 | **0.263** ± 0.017 | 0.892 ± 0.019 | 1% |
| B | yes | synthetic | 249 | 0.132 ± 0.011 | 0.877 ± 0.020 | 0% |
| C | yes | natural | 579 | 0.198 ± 0.039 | 0.571 ± 0.023 | 61% |
| D | yes | synthetic + natural | 829 | 0.164 ± 0.021 | 0.833 ± 0.025 | 57% |

Macro F1, mean ± sd over seeds. True Cooperative rate on `test_natural` is 0%.

### 9.1 No configuration beats chance on natural data

Binomial test against uniform guessing over the four classes present
(H₀: accuracy = 0.25):

| cfg | mean accuracy | p |
|---|---|---|
| A | 0.307 | 0.252 — ns |
| B | 0.233 | 0.618 — ns |
| C | 0.140 | 0.981 — ns |
| D | 0.127 | 0.993 — ns |

The best transformer configuration is 0.263 macro F1 where TF-IDF reached 0.254
and drawing from the class priors reaches 0.212. Fine-tuning roberta-base buys
nothing measurable on naturally occurring data.

### 9.2 On synthetic data it works, and that is the contrast

0.83–0.89 macro F1 across every synthetic-trained config, stable across seeds.
Against the illiterate surface baseline's 0.599, RoBERTa is clearly extracting
real lexical signal here. The same model, same weights, same label scheme, moved
to real conversation: 0.26.

That gap — 0.89 synthetic vs 0.26 natural — is the result worth publishing.

### 9.3 What context actually does (item 3)

A → B is the only change of context, and macro F1 halves (0.263 → 0.132). But
that overstates it: accuracy moves 0.31 → 0.23, and exact McNemar per seed gives
p = 0.39, 0.79, 0.125. **Not significant.** With 50 items there is no resolving
power for a difference this size.

What actually changes is the *shape* of the predictions. Pooled over seeds on
`test_natural`:

| cfg | predicted label distribution |
|---|---|
| A | Quality 53%, Relation 37%, Manner 5%, Quantity 4%, Coop 1% |
| B | **Quality 92%**, Relation 7%, Manner 1% |
| C | **Cooperative 61%**, Relation 17%, Quality 15%, Manner 5%, Quantity 3% |
| D | **Cooperative 57%**, Quality 23%, Relation 15%, Quantity 5%, Manner 1% |
| gold | Quantity 30%, Manner 24%, Relation 24%, Quality 22% |

Adding context makes the model collapse onto a single class out of domain.
Macro F1 punishes that heavily while accuracy barely registers it, which is why
the two metrics disagree.

This is not a truncation artefact. Only 18% of natural pairs exceed the 128-token
window and no utterance exceeds it alone. Splitting the test set: on the 41
untruncated items A beats B by +0.163 macro F1; on the 9 truncated items the gap
is −0.012. The effect is largest exactly where truncation is absent.

### 9.4 Training domain decides which class it collapses to

- Synthetic-trained (A, B) → collapse onto Quality/Relation, ~0–1% Cooperative
- Natural-trained (C, D) → collapse onto Cooperative, 57–61%

Same architecture, same hyperparameters. The training register picks the
attractor. C and D score *below* chance on natural data precisely because the
Cooperative prior they learned is wrong for every item in this test set.

Per-class detail for the strongest config (A, seed 0) shows the pattern that
macro F1 is summarising — high precision, near-zero recall on the classes it
avoids:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Quantity | 1.00 | 0.13 | 0.24 | 15 |
| Relation | 0.32 | 0.58 | 0.41 | 12 |
| Quality | 0.25 | 0.55 | 0.34 | 11 |
| Manner | 0.50 | 0.08 | 0.14 | 12 |

### 9.5 Caveat that limits all of §9

`test_natural` has no Cooperative items (§6), so configs C and D are penalised
on every Cooperative prediction they make. Their natural-domain numbers are a
lower bound and the comparison against A and B is not clean. Annotating
`test_natural_pending.csv` is what fixes this; until then, treat §9.4's direction
as established and its magnitude as not.

## 10. Error analysis (item 11)

`src/error_analysis.py`, over all 12 ablation runs. Causes are split into what
the numbers determine and what only a reading of the item can determine; the
script computes the first and emits coding sheets for the second.

### 10.1 Nothing in the natural set is reliably classified

| | test_natural | test_synthetic |
|---|---|---|
| items every run gets right | **0 / 50 (0%)** | 24 / 73 (33%) |
| items no run ever gets right | **24 / 50 (48%)** | 2 / 73 (3%) |

There is no item in the natural test set that all twelve configurations agree on
and get right. Half of it is wrong under every configuration tried. Whatever the
0.26 macro F1 is measuring, it is not a stable core of items the approach
handles — it is scattered partial credit.

### 10.2 Recall collapses class by class, and the biggest class fares worst

Pooled over all runs:

| class | natural recall | synthetic recall | share of natural gold |
|---|---|---|---|
| Quantity | **6.1%** | 95.8% | 30% |
| Manner | **8.3%** | 82.1% | 24% |
| Relation | 25.0% | 78.6% | 24% |
| Quality | 47.0% | 65.7% | 22% |

Quantity is the largest class in the natural test set and is recovered 6% of the
time. On synthetic it is the *easiest* class at 96%. The ordering of difficulty
does not merely weaken across domains — it inverts.

### 10.3 Where the mass goes

Most common gold → modal-prediction confusions on natural:

| | count |
|---|---|
| Manner → Quality | 7 |
| Quantity → Cooperative | 7 |
| Relation → Quality | 6 |
| Quantity → Quality | 5 |
| Quality → Cooperative | 4 |

Two sinks: **Quality** for synthetic-trained configs and **Cooperative** for
natural-trained ones. That is §9.4's attractor effect at the item level — the
errors are not diffuse, they drain into whichever class the training register
favours.

### 10.4 Mechanical failure flags (natural, n=50)

| flag | items | share |
|---|---|---|
| systematically-missed-class | 27 | 54% |
| never-correct | 24 | 48% |
| collapse-to-modal-class | 24 | 48% |
| config-dependent | 23 | 46% |
| context-dominates-length | 13 | 26% |
| truncated-at-128 | 9 | 18% |

Nearly half of all natural failures are the model emitting its fallback class.
On synthetic the dominant flag is instead `config-dependent` (42%) with almost
no collapse — the failures there are ordinary seed and regime variance, which is
what failure looks like when a model is actually working.

### 10.5 The judgmental half is not done, on purpose

`ambiguous_annotation`, `sarcasm`, `multiple_maxims`, `annotation_error` and
`insufficient_context` are readings of an item, not properties of a prediction.
Asserting them from a script would manufacture the finding the checklist is
asking you to *measure*. Two sheets are written instead, covering the 99 items
that any run got wrong:

- `results/gold_recheck_sheet.csv` — item and gold label only, **no model
  output**. Answers "is this gold label defensible?" Showing a prediction here
  would contaminate the answer.
- `results/error_coding_sheet.csv` — item, gold, per-config predictions, and
  the mechanical flags. Answers "why did the model fail?"

Do the blind sheet first. The quantified breakdown item 11 asks for comes from
coding these, and given §10.1 the honest prior is that a meaningful share of the
50 will turn out to be annotation disagreement rather than model failure — which
is exactly why item 2's agreement study has to land before these numbers get a
final interpretation.

## 11. Not yet done

- Item 8's RoBERTa ablation grid — the cheap models already show the shape, but
  the transformer numbers are what the paper claims.
- Items 5, 6, 9, 11 — hard-phenomenon subsets, minimal pairs, task
  decomposition, 100-failure error analysis.
- `README.md` still reports macro F1 0.91 and 0.77 from splits that included the
  gold items and had no provenance separation. Those numbers need re-deriving
  against `corpus_train.csv` before they are quoted anywhere.
