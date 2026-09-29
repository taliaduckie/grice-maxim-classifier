"""Run Claude as one annotator over an annotation sheet.

Writes the same file annotate.py writes for a human, so agreement.py and
finalize_test_set.py treat the model as one more annotator. The model sees
exactly what a human annotator sees — the guidelines and the (context,
utterance) pair — and nothing else: no subreddit, no gold label, no model
prediction. Output is never written into the sheet itself, so it cannot
anchor a human pass.

    python3 src/llm_annotate.py                        # natural_qa_pairs -> data/test/annotations/<model>.csv
    python3 src/llm_annotate.py --limit 5              # smoke test
    python3 src/llm_annotate.py --sheet results/gold_recheck_sheet.csv \
                                --out results/gold_recheck_<model>.csv

Resumable: rows already present in --out are skipped.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from annotate import (
    MAXIM_CHOICES, VTYPE_CHOICES, append_annotation, done_ids, load_sheet,
)
from paths import DOCS_DIR, QA_PAIRS_PATH, TEST_DIR

GUIDELINES_PATH = DOCS_DIR / "annotation_guidelines.md"
MODEL_DEFAULT = "claude-opus-5-5"

FRAMING = (
    "You are one annotator in an inter-annotator agreement study of Gricean "
    "maxim violations in Reddit exchanges. Follow the annotation guidelines "
    "below exactly, including the decision ordering in section 2, the "
    "flouting test in section 3, the fixed conventions in section 5, and the "
    "instruction that Cooperative is a real label and the expected answer for "
    "most natural conversation. You see only the pair; do not guess at the "
    "subreddit or the speaker's private situation. Use `unknown` and "
    "`unlabelable` freely where the guidelines say to. Notes are required for "
    "confidence 1-3 and for `clash`; keep them to one sentence. If a second "
    "maxim also applies, write it in notes as `also: <Maxim>` (section 6).\n\n"
    "=== ANNOTATION GUIDELINES ===\n\n"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "maxim": {"type": "string", "enum": MAXIM_CHOICES},
        "violation_type": {"type": "string", "enum": VTYPE_CHOICES + ["none", ""]},
        "confidence_1_to_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "notes": {"type": "string"},
    },
    "required": ["maxim", "violation_type", "confidence_1_to_5", "notes"],
    "additionalProperties": False,
}


def system_prompt():
    return FRAMING + GUIDELINES_PATH.read_text(encoding="utf-8")


def item_prompt(row):
    return (
        f"context (the prior turn):\n{row['context']}\n\n"
        f"utterance (the response to label):\n{row['utterance']}"
    )


def label_one(client, model, effort, system, row):
    """Return (row, judgment dict) or (row, None) on refusal / error."""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": item_prompt(row)}],
        )
    except anthropic.APIError as e:
        print(f"  {row['row_id']}: {type(e).__name__}: {e}", file=sys.stderr)
        return row, None
    if resp.stop_reason == "refusal":
        cat = resp.stop_details.category if resp.stop_details else None
        print(f"  {row['row_id']}: refused ({cat})", file=sys.stderr)
        return row, None
    if resp.stop_reason != "end_turn":
        print(f"  {row['row_id']}: stop_reason={resp.stop_reason}", file=sys.stderr)
        return row, None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return row, json.loads(text)
    except json.JSONDecodeError:
        print(f"  {row['row_id']}: unparseable output", file=sys.stderr)
        return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default=str(QA_PAIRS_PATH))
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--effort", default="medium",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--out", default=None,
                    help="default data/test/annotations/<model>.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="label only the first N pending rows")
    args = ap.parse_args()

    out = Path(args.out) if args.out else TEST_DIR / "annotations" / f"{args.model}.csv"
    rows = load_sheet(args.sheet)
    done = done_ids(out)
    pending = [r for r in rows if r["row_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(rows)} rows in {args.sheet}; {len(done)} already in {out}; "
          f"labelling {len(pending)} with {args.model} (effort={args.effort})")
    if not pending:
        return

    client = anthropic.Anthropic()
    system = system_prompt()
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(label_one, client, args.model, args.effort, system, r)
                for r in pending]
        for i, fut in enumerate(as_completed(futs), 1):
            row, judgment = fut.result()
            results[row["row_id"]] = judgment
            if i % 20 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)}")

    # Write in sheet order so the file diffs cleanly across resumed runs.
    written = skipped = 0
    for row in pending:
        j = results.get(row["row_id"])
        if j is None:
            skipped += 1
            continue
        try:
            append_annotation(out, row, j["maxim"], j["violation_type"],
                              j["confidence_1_to_5"], j["notes"])
            written += 1
        except ValueError as e:
            print(f"  {row['row_id']}: rejected: {e} — {j}", file=sys.stderr)
            skipped += 1
    print(f"wrote {written} to {out}; {skipped} skipped (re-run to retry them)")


if __name__ == "__main__":
    main()
