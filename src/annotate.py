"""Annotation UI for the natural QA pairs and the pending test sheet.

Walks one pair at a time, enforces the label vocabulary, and writes the exact
format agreement.py reads. Exists because the last hand-annotated file arrived
with shifted columns, out-of-vocabulary labels, and short forms — a spreadsheet
can't stop any of that, and this can.

    python3 src/annotate.py --annotator talia
    python3 src/annotate.py --annotator talia --sheet data/test/test_natural_pending.csv

Rules from docs/annotation_guidelines.md that the UI enforces:
  - Cooperative forces violation_type "none" (§3)
  - Relation defaults to "unknown" (§3 standing rule)
  - no model output is shown anywhere (§7, the ed9a122 anchoring incident)
  - items are presented in a per-annotator shuffled order, so two annotators
    never see the same thread back-to-back in the same sequence (§9)

Progress is saved after every item; rerunning with the same --annotator resumes
where you left off.
"""

import argparse
import csv
import random
from pathlib import Path

from labels import MAXIMS, VIOLATION_TYPES
from paths import QA_PAIRS_PATH, TEST_DIR

ANNOTATIONS_DIR = TEST_DIR / "annotations"

MAXIM_CHOICES = MAXIMS + ["unlabelable"]
VTYPE_CHOICES = [v for v in VIOLATION_TYPES if v != "none"]

OUT_COLUMNS = ["row_id", "context", "utterance", "maxim", "violation_type",
               "confidence_1_to_5", "notes"]

# §2's ordering, shown as a permanent reminder next to the item.
PROCEDURE = (
    "Decision order (stop at the first that applies):  "
    "0 not a contribution → unlabelable ·  "
    "1 off-topic → Relation ·  2 false/unsupported → Quality ·  "
    "3 too little/too much → Quantity ·  4 unclear in form → Manner ·  "
    "5 none of those → Cooperative"
)


def load_sheet(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ("row_id", "context", "utterance") if c not in rows[0]]
    if missing:
        raise SystemExit(f"{path} lacks columns {missing} — not an annotation sheet")
    return rows


def done_ids(out_path):
    if not out_path.exists():
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {r["row_id"] for r in csv.DictReader(f) if r.get("maxim")}


def work_order(rows, annotator, already_done):
    """Per-annotator deterministic shuffle, minus finished items.

    Shuffling breaks up thread blocks (items from one thread otherwise arrive
    consecutively and share the annotator's sequential attention), and seeding
    by name gives each annotator a different order while keeping each one's
    order stable across sessions.
    """
    remaining = [r for r in rows if r["row_id"] not in already_done]
    rng = random.Random(f"grice-{annotator}")
    rng.shuffle(remaining)
    return remaining


def append_annotation(out_path, row, maxim, vtype, confidence, notes):
    """Validate and append one judgment. Raises ValueError on schema breaks."""
    if maxim not in MAXIM_CHOICES:
        raise ValueError(f"maxim must be one of {MAXIM_CHOICES}")
    if maxim == "Cooperative":
        vtype = "none"                      # §3: none is reserved for Cooperative
    elif maxim == "unlabelable":
        vtype = ""
    elif vtype not in VTYPE_CHOICES:
        raise ValueError(f"violation_type must be one of {VTYPE_CHOICES}")
    if str(confidence) not in {"1", "2", "3", "4", "5"}:
        raise ValueError("confidence must be 1-5")
    if maxim == "clash" or vtype == "clash":
        pass  # clash is legal; §4 asks for a note, checked below
    if vtype == "clash" and not notes.strip():
        raise ValueError("clash needs a sentence in notes (§4)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, quoting=csv.QUOTE_ALL)
        if is_new:
            w.writeheader()
        w.writerow({
            "row_id": row["row_id"],
            "context": row["context"],
            "utterance": row["utterance"],
            "maxim": maxim,
            "violation_type": vtype,
            "confidence_1_to_5": str(confidence),
            "notes": notes.strip(),
        })


def default_vtype(maxim):
    if maxim == "Cooperative":
        return "none"
    if maxim == "Relation":
        return "unknown"                    # §3 standing rule
    return None


def build_ui(rows, out_path, annotator):
    import gradio as gr

    state = {"queue": work_order(rows, annotator, done_ids(out_path))}
    total = len(rows)

    def current():
        if not state["queue"]:
            return None
        return state["queue"][0]

    def render():
        row = current()
        n_done = total - len(state["queue"])
        progress = f"{n_done} / {total}"
        if row is None:
            return ("**All items annotated.** Run "
                    "`python3 src/agreement.py data/test/annotations/*.csv` "
                    "once a second annotator finishes.", "", progress)
        return (f"**Q (context):** {row['context']}",
                f"**A (utterance):** {row['utterance']}", progress)

    def submit(maxim, vtype, confidence, notes):
        row = current()
        if row is None:
            return (*render(), gr.update(), "")
        try:
            append_annotation(out_path, row, maxim, vtype or "", int(confidence), notes)
        except ValueError as e:
            return (*render(), gr.update(), f"Not saved: {e}")
        state["queue"].pop(0)
        return (*render(), gr.update(value=None), "")

    def on_maxim(maxim):
        return gr.update(value=default_vtype(maxim),
                         interactive=maxim not in ("Cooperative", "unlabelable"))

    with gr.Blocks(title=f"Maxim annotation — {annotator}") as demo:
        gr.Markdown(f"### Annotating as **{annotator}** → `{out_path}`")
        gr.Markdown(PROCEDURE)
        context_md = gr.Markdown()
        utterance_md = gr.Markdown()
        progress_md = gr.Markdown()
        maxim_in = gr.Radio(MAXIM_CHOICES, label="Maxim (§2 order)")
        vtype_in = gr.Radio(VTYPE_CHOICES, label="Violation type (§3)")
        conf_in = gr.Slider(1, 5, value=3, step=1,
                            label="Confidence (1-2 routes to adjudication)")
        notes_in = gr.Textbox(label="Notes (required for clash; use for 'also: <maxim>')")
        error_md = gr.Markdown()
        save_btn = gr.Button("Save and next", variant="primary")

        maxim_in.change(on_maxim, maxim_in, vtype_in)
        save_btn.click(submit, [maxim_in, vtype_in, conf_in, notes_in],
                       [context_md, utterance_md, progress_md, maxim_in, error_md])
        demo.load(lambda: render(), None, [context_md, utterance_md, progress_md])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", required=True,
                    help="Your name; output goes to data/test/annotations/<name>.csv")
    ap.add_argument("--sheet", default=str(QA_PAIRS_PATH),
                    help="Sheet to annotate (default: the 187 natural QA pairs)")
    args = ap.parse_args()

    rows = load_sheet(Path(args.sheet))
    out_path = ANNOTATIONS_DIR / f"{args.annotator}.csv"
    remaining = len(work_order(rows, args.annotator, done_ids(out_path)))
    print(f"{len(rows)} items in sheet, {len(rows) - remaining} done, "
          f"{remaining} to go.")
    demo = build_ui(rows, out_path, args.annotator)
    demo.launch()


if __name__ == "__main__":
    main()
