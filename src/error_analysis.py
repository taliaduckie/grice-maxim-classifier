"""Error analysis over the saved ablation predictions (checklist item 11).

The checklist asks for ~100 failures categorised by cause. Those causes split
into two kinds, and conflating them would produce a fabricated result:

  mechanical   determinable from the predictions and the text alone — did the
               model fall back on its modal class, was the pair truncated, do
               all twelve runs fail on this item, is the whole gold class being
               missed. This script computes these.

  judgmental   "ambiguous annotation", "sarcasm", "multiple maxims",
               "annotation error". These are readings of the item. A script
               asserting them would be guessing. This script emits coding
               sheets instead.

Two sheets, deliberately separate:

  gold_recheck_sheet.csv   item + gold label, NO model output. Answers "is the
                           gold label defensible?" Showing a prediction here
                           would contaminate the judgment — you cannot ask
                           someone whether a label is wrong while showing them
                           what a model guessed instead.
  error_coding_sheet.csv   item + gold + predictions + mechanical flags.
                           Answers "why did the model fail?" Predictions are
                           required for this one.

Do the gold recheck first, and do it before reading the second sheet.

Usage:
    python3 src/error_analysis.py
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from ablations import CONFIGS, TEST_NATURAL, TEST_SYNTHETIC, load
from labels import MAXIMS

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
ABLATIONS = RESULTS_DIR / "ablations.json"

MAX_TOKENS = 128
SHEET_SIZE = 100


def token_lengths(rows):
    """Token counts under the same tokenizer the models used."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("roberta-base")
    utt = [len(tok(r["utterance"])["input_ids"]) for r in rows]
    ctx = [len(tok(r["context"])["input_ids"]) for r in rows]
    return utt, ctx


def collect(split_name, rows, runs):
    """Per-item prediction record across every config and seed."""
    by_run = {}
    for r in runs:
        preds = r["splits"][split_name]["predictions"]
        ids = r["splits"][split_name]["row_ids"]
        by_run[(r["config"], r["seed"])] = dict(zip(ids, preds))

    # each config's modal prediction on this split — its fallback class
    modal = {}
    for cfg in CONFIGS:
        counts = Counter()
        for (c, s), table in by_run.items():
            if c == cfg:
                counts.update(table.values())
        modal[cfg] = counts.most_common(1)[0][0] if counts else None

    utt_len, ctx_len = token_lengths(rows)
    items = []
    for i, row in enumerate(rows):
        rid = row["row_id"]
        preds = {k: v[rid] for k, v in by_run.items() if rid in v}
        correct = {k: (p == row["maxim"]) for k, p in preds.items()}
        by_config = defaultdict(list)
        for (cfg, seed), p in preds.items():
            by_config[cfg].append(p)

        n_runs = len(preds)
        n_correct = sum(correct.values())
        pred_counts = Counter(preds.values())
        items.append({
            "row_id": rid,
            "utterance": row["utterance"],
            "context": row["context"],
            "gold": row["maxim"],
            "gold_violation_type": row.get("violation_type", ""),
            "subreddit": row.get("subreddit", ""),
            "n_runs": n_runs,
            "n_correct": n_correct,
            "accuracy": n_correct / n_runs if n_runs else 0.0,
            "modal_prediction": pred_counts.most_common(1)[0][0] if pred_counts else "",
            "prediction_entropy_labels": len(pred_counts),
            "by_config": {c: Counter(v).most_common(1)[0][0] for c, v in by_config.items()},
            "utt_tokens": utt_len[i],
            "ctx_tokens": ctx_len[i],
            "truncated": utt_len[i] + ctx_len[i] > MAX_TOKENS,
            "modal_by_config": modal,
        })
    return items, modal


def mechanical_flags(item, class_recall):
    """Causes that follow from the numbers, not from a reading of the text."""
    flags = []
    if item["n_correct"] == 0:
        flags.append("never-correct")
    elif item["accuracy"] < 0.5:
        flags.append("mostly-wrong")

    # did each config simply emit its fallback class here?
    fell_back = [c for c, p in item["by_config"].items()
                 if p == item["modal_by_config"][c] and p != item["gold"]]
    if len(fell_back) >= 3:
        flags.append("collapse-to-modal-class")
    elif fell_back:
        flags.append(f"collapse-in-{','.join(sorted(fell_back))}")

    # correct under some training regimes but not others
    right_cfgs = {c for c, p in item["by_config"].items() if p == item["gold"]}
    if right_cfgs and len(right_cfgs) < len(item["by_config"]):
        flags.append("config-dependent")

    if item["truncated"]:
        flags.append("truncated-at-128")

    if class_recall.get(item["gold"], 1.0) < 0.15:
        flags.append("systematically-missed-class")

    if item["ctx_tokens"] > 3 * max(item["utt_tokens"], 1):
        flags.append("context-dominates-length")

    return flags


def per_class_recall(items):
    hits, total = Counter(), Counter()
    for it in items:
        total[it["gold"]] += it["n_runs"]
        hits[it["gold"]] += it["n_correct"]
    return {c: hits[c] / total[c] for c in total if total[c]}


def write_sheet(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def report(split_name, items, modal):
    print(f"\n{'='*72}\n{split_name.upper()}  ({len(items)} items x "
          f"{items[0]['n_runs']} runs)\n{'='*72}")

    recall = per_class_recall(items)
    print("Per-class recall pooled over all runs:")
    for cls, r in sorted(recall.items(), key=lambda kv: kv[1]):
        bar = "#" * int(r * 40)
        print(f"  {cls:<13}{r:>6.1%}  {bar}")

    print("\nEach config's fallback (modal) class on this split:")
    for cfg, m in modal.items():
        print(f"  {cfg}: {m}")

    n_never = sum(1 for it in items if it["n_correct"] == 0)
    n_always = sum(1 for it in items if it["n_correct"] == it["n_runs"])
    print(f"\nItems no run ever gets right : {n_never}/{len(items)} "
          f"({n_never/len(items):.0%})")
    print(f"Items every run gets right   : {n_always}/{len(items)} "
          f"({n_always/len(items):.0%})")

    flag_counts = Counter()
    for it in items:
        for f in it["flags"]:
            flag_counts[f] += 1
    print("\nMechanical failure flags (items may carry several):")
    for f, n in flag_counts.most_common():
        print(f"  {f:<32}{n:>5}  ({n/len(items):.0%})")

    print("\nMost common gold -> modal-prediction confusions:")
    conf = Counter((it["gold"], it["modal_prediction"]) for it in items
                   if it["modal_prediction"] != it["gold"])
    for (g, p), n in conf.most_common(8):
        print(f"  {g:<13} -> {p:<13}{n:>4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablations", default=str(ABLATIONS))
    ap.add_argument("--sheet-size", type=int, default=SHEET_SIZE)
    args = ap.parse_args()

    path = Path(args.ablations)
    if not path.exists():
        print(f"No ablation results at {path}. Run src/ablations.py first.",
              file=sys.stderr)
        return 1
    runs = json.loads(path.read_text())["runs"]

    splits = {"natural": load(TEST_NATURAL), "synthetic": load(TEST_SYNTHETIC)}
    all_items = {}
    for split_name, rows in splits.items():
        items, modal = collect(split_name, rows, runs)
        recall = per_class_recall(items)
        for it in items:
            it["flags"] = mechanical_flags(it, recall)
            it["split"] = split_name
        report(split_name, items, modal)
        all_items[split_name] = items

    # ---- the two coding sheets --------------------------------------------
    pool = [it for items in all_items.values() for it in items
            if it["n_correct"] < it["n_runs"]]
    pool.sort(key=lambda it: (it["accuracy"], it["row_id"]))
    selected = pool[:args.sheet_size]

    gold_rows = [{
        "row_id": it["row_id"],
        "split": it["split"],
        "context": it["context"],
        "utterance": it["utterance"],
        "current_gold_maxim": it["gold"],
        "current_gold_violation_type": it["gold_violation_type"],
        "gold_defensible_y_n_unsure": "",
        "better_maxim_if_no": "",
        "is_ambiguous_between": "",
        "notes": "",
    } for it in selected]

    error_rows = [{
        "row_id": it["row_id"],
        "split": it["split"],
        "context": it["context"],
        "utterance": it["utterance"],
        "gold": it["gold"],
        "runs_correct": f"{it['n_correct']}/{it['n_runs']}",
        "pred_A_no_context": it["by_config"].get("A", ""),
        "pred_B_context": it["by_config"].get("B", ""),
        "pred_C_natural": it["by_config"].get("C", ""),
        "pred_D_both": it["by_config"].get("D", ""),
        "mechanical_flags": ";".join(it["flags"]),
        "utt_tokens": it["utt_tokens"],
        "ctx_tokens": it["ctx_tokens"],
        "sarcasm_or_irony_y_n": "",
        "multiple_maxims_y_n": "",
        "needs_more_context_y_n": "",
        "primary_cause": "",
        "notes": "",
    } for it in selected]

    write_sheet(RESULTS_DIR / "gold_recheck_sheet.csv", gold_rows, list(gold_rows[0]))
    write_sheet(RESULTS_DIR / "error_coding_sheet.csv", error_rows, list(error_rows[0]))

    print(f"\n{'='*72}\nCODING SHEETS\n{'='*72}")
    print(f"{len(selected)} hardest failures selected from {len(pool)} imperfect items.")
    print(f"  results/gold_recheck_sheet.csv   — blind. Do this one FIRST.")
    print(f"  results/error_coding_sheet.csv   — shows predictions. Do this SECOND.")
    print("\nAllowed values for primary_cause:")
    for cause in ["insufficient_context", "ambiguous_annotation", "sarcasm",
                  "multiple_maxims", "lexical_shortcut", "domain_shift",
                  "genuine_model_failure", "annotation_error"]:
        print(f"  {cause}")
    print("\nThe mechanical_flags column is evidence, not the answer — a "
          "collapse-to-modal-class\nitem can still be an annotation error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
