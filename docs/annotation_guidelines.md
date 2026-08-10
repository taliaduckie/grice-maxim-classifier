# Annotation guidelines — Gricean maxim labelling

Version 1.0 · for use with `data/test/test_natural_pending.csv`

These guidelines exist so that two people looking at the same exchange reach the
same label for the same stated reason. If they can't, the label is not a fact
about the exchange and no classifier trained on it means anything.

Read all of §1–§4 before labelling anything. Do the calibration set in §8 first.

---

## 1. The unit of annotation

Every item is a **pair**:

- **context** — the prior turn (a question, a claim, a request)
- **utterance** — the response being labelled

You are labelling the **utterance**, judged *relative to the context*. Never
label an utterance on its own. If the context is missing or unintelligible,
mark `maxim = unlabelable` and give the reason in `notes`.

You are judging the utterance **as a conversational move**, not its content. A
response can be rude, wrong, boring, or morally objectionable and still be
perfectly cooperative in Grice's sense. Cooperation is about informational fit,
not niceness.

---

## 2. The decision procedure

Work through these in order. **Stop at the first one that applies.** The order
is not arbitrary — it exists to break ties that would otherwise be a coin flip.

### Step 0 — Is it a conversational contribution at all?

If the utterance is not a response to the context (a bot post, a deleted
comment, a link with no text, a reply to a different comment), mark
`unlabelable`. Do not force it.

### Step 1 — Relation

> Does the utterance address the question or topic raised by the context?

If it does not — if a listener would have to do interpretive work to see what it
has to do with the context at all — the label is **Relation**.

Relation outranks the others because an off-topic response cannot be assessed
for informativeness or truth *with respect to a question it isn't answering*.

**Not Relation:** a response that answers *indirectly* but recoverably.
"Is there a gas station nearby?" / "There's one at the next exit" is
Cooperative. "Is there a gas station nearby?" / "I'm going to be late" is
Relation.

### Step 2 — Quality

> Is the utterance false, or asserted without adequate evidence?

If yes, the label is **Quality**. This covers both plain falsehood and the
non-literal cases (irony, sarcasm, hyperbole, metaphor) where the literal
content is false and the speaker knows the listener knows.

**Requires evidence.** You may only label Quality when the falsity is
recoverable from the pair itself or from common knowledge. If judging it would
require you to know facts about the speaker's private situation, it is **not**
Quality — use `unlabelable` or fall through to Step 3.

### Step 3 — Quantity

> Does the utterance give too little or too much information for the context?

**Too little:** the context poses a question with a determinate answer and the
utterance underdelivers. "Did the client sign?" / "The client seemed interested
in the terms." The question was yes/no; the answer is neither.

**Too much:** the utterance supplies substantially more than the context called
for, in a way that obscures the answer. Length alone is not enough — a long
answer to a question that genuinely needs a long answer is Cooperative.

**Boundary with Manner:** if the extra material makes it *hard to find* the
answer, that is Manner. If the extra material is *irrelevant but clear*, that is
Quantity. Ask: "was the answer hard to extract, or just accompanied?"

### Step 4 — Manner

> Is the utterance unclear, ambiguous, disorganised, or needlessly obscure?

Manner is about **form**, not content. The information may all be there and be
true and be the right amount — but delivered so that the listener struggles.

Manner is the residual category and the most over-applied. Do not use it for
informal register, typos, profanity, or dialect. Reddit prose is not a Manner
violation for being Reddit prose. Ask: "would a cooperative listener have to
re-read this to work out what's meant?"

### Step 5 — Cooperative

None of the above applied. The utterance is relevant, adequately supported,
appropriately informative, and clear.

**Cooperative is a real label, not a fallback.** It is the correct answer for
most naturally occurring conversation. Do not reach for a violation because an
item "must be interesting" — the sample is not curated to be interesting.

---

## 3. Flouting vs. violating

Only assign a `violation_type` once you have assigned a non-Cooperative maxim.

| Value | Use when |
|---|---|
| `none` | The maxim label is Cooperative. |
| `flouting` | The breach is **blatant and intended to be noticed**, and the listener is meant to derive a further meaning from it. |
| `violating` | The breach is real but **not designed to be seen** — the speaker is failing, misleading, or careless. |
| `opting_out` | The speaker **overtly declines** to contribute: "no comment", "I'd rather not say", "can't discuss that". |
| `clash` | Satisfying one maxim would have required breaking another, and the speaker visibly chose. Rare; justify in `notes`. |
| `unknown` | The evidence in the pair does not settle it. **Use this freely.** |

**The operative test for flouting:** would the speaker expect the listener to
*notice* the breach and draw a conclusion from it? Sarcasm passes ("Great, another
meeting" — the speaker wants you to see the falsity). A vague status update to
avoid admitting slippage does not: it is designed to slide past unnoticed. That
is `violating`.

**Standing rule for Relation.** Default Relation items to `unknown` unless there
is explicit evidence of intent in the text. Coherence scoring on this corpus
(see `src/score_corpus.py` and the note in `src/labels.py`) found deliberate
deflection and genuine irrelevance to be statistically indistinguishable at the
surface: mean L = 0.094 vs 0.049, within one standard deviation. Annotators
guessing intent here are producing noise, not signal.

---

## 4. Confidence and notes

- `confidence_1_to_5` — how sure you are. **1 or 2 is a legitimate answer.**
  Low-confidence items are routed to adjudication rather than averaged away.
- `notes` — for any item scored 1–3, and for every `clash`: one sentence on what
  the competing reading was. This is what makes disagreements resolvable.

---

## 5. Cases with a fixed convention

These recur and would otherwise be split 50/50 between annotators. The
convention is stipulated; follow it even where you'd have chosen otherwise, and
record your objection in `notes`.

| Case | Label | Why |
|---|---|---|
| "No comment" / "I'd rather not say" | Quantity / `opting_out` | Underinformative, but the move is overt refusal, not implicature. |
| Sarcasm ("Oh, brilliant") | Quality / `flouting` | Literal content false, breach meant to be seen. |
| Hyperbole ("told you a million times") | Quality / `flouting` | Same structure as sarcasm. |
| Rhetorical question as answer | Relation / `flouting` | Only if it doesn't answer; if it answers by implication, Cooperative. |
| Answering a different-but-adjacent question | Relation / `unknown` | Adjacency is not relevance. |
| Long anecdote that does contain the answer | Cooperative | Length is not a violation on its own. |
| Long anecdote that buries the answer | Manner | Extraction cost is the deciding factor. |
| Joke reply that is also on-topic | Cooperative | Register is not a violation. |
| Joke reply that displaces the answer | Relation / `flouting` | The joke is doing the work instead of the answer. |
| Confident wrong factual claim | Quality / `violating` | Not flouting: it isn't meant to be noticed. |
| Hedged claim ("I think, not sure") | Cooperative | Hedging *satisfies* Quality — it marks the evidence level. |
| Profanity, slang, informal register | Not a violation by itself | Fall through to Step 5. |

---

## 6. Multiple maxims

If two maxims genuinely apply, record the one reached first by §2's ordering in
`maxim`, and write the second in `notes` as `also: Manner`. The current
single-label schema cannot represent both, and pretending otherwise loses the
information. Counting how often this happens is itself a result worth reporting
(checklist item 5).

---

## 7. What not to do

- Do **not** look at any model prediction before labelling. This corpus already
  has one anchoring incident on record (CHANGELOG, `ed9a122`): a default value
  shown to annotators propagated into the labels.
- Do **not** discuss items with the other annotator during the independent pass.
  Agreement measured after discussion measures the discussion.
- Do **not** revise earlier items after developing a theory mid-pass. Note the
  theory; the adjudication round is where it gets applied.
- Do **not** avoid `unknown` or `unlabelable` to make the sheet look complete.
  A forced label is worse than a missing one.

---

## 8. Calibration set

Before the independent pass, label these ten by hand and compare against the
stated answers. If you disagree with an answer, the guideline is wrong or
underspecified — say so and amend it *before* annotating for real.

| # | Context | Utterance | Answer |
|---|---|---|---|
| 1 | "Why were you late?" | "The weather is nice today." | Relation / `unknown` |
| 2 | "Did you finish the report?" | "I've been swamped all week." | Quantity / `violating` |
| 3 | "How was the presentation?" | "Oh, it was a triumph." (it wasn't) | Quality / `flouting` |
| 4 | "Where's the nearest station?" | "Two blocks north, past the church." | Cooperative |
| 5 | "Can you explain the outage?" | "There was a thing with the thing, and then it resolved itself, mostly." | Manner / `violating` |
| 6 | "Are you seeing anyone?" | "I'd rather not get into it." | Quantity / `opting_out` |
| 7 | "Is the build passing?" | "It's green." | Cooperative |
| 8 | "What's your salary?" | "Enough." | Quantity / `flouting` |
| 9 | "Did you take the last coffee?" | "There was coffee?" | Relation / `unknown` |
| 10 | "How long will the migration take?" | "Somewhere between a day and a quarter, depending on scope, though scope isn't settled, and there are dependencies." | Manner / `unknown` |

Target: at least 8/10 on the maxim before proceeding. Violation-type agreement
will be lower and that is expected — it is the harder judgment.

---

## 9. Workflow

1. Each annotator copies `data/test/test_natural_pending.csv` to
   `data/test/annotations/<yourname>.csv` and fills `maxim`,
   `violation_type`, `confidence_1_to_5`, `notes`. Leave `row_id` untouched.
2. No discussion until every annotator has finished.
3. Run `python3 src/agreement.py data/test/annotations/*.csv` for the agreement
   report and the adjudication sheet.
4. Adjudicate: for each disagreement, both annotators state their reading; a
   third party decides, or the item goes to `unknown`. Record the outcome and
   the reason.
5. Any convention invented during adjudication is added to §5 and the version
   number here goes up. Guidelines are an output of annotation, not just an
   input.
