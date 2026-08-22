"""Inter-annotator agreement and adjudication.

    python3 src/agreement.py data/test/annotations/*.csv

    percent agreement  raw match rate; inflated when one label dominates
    Cohen / Fleiss k   chance-corrected; Cohen for 2 annotators, Fleiss for 3+.
                       Both need every item labelled by everyone.
    Krippendorff a     chance-corrected, tolerates missing values and any number
                       of annotators. Report this one when annotators skipped
                       items, which the guidelines tell them to do.

Landis & Koch (1977) rule of thumb: <0.20 slight, 0.21-0.40 fair, 0.41-0.60
moderate, 0.61-0.80 substantial, >0.80 almost perfect. Below ~0.4 on a 5-way
pragmatic judgment means the scheme isn't measuring anything yet.
"""

import argparse
import csv
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from labels import MAXIMS, VIOLATION_TYPES

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "test" / "adjudication_sheet.csv"

# Labels an annotator may use that aren't in the primary schema.
EXTRA_MAXIM_LABELS = ["unlabelable"]

MISSING = {"", "-", "na", "n/a", "none given", "?"}


def load_annotations(paths, field):
    """{annotator: {row_id: label}} plus the shared item text.

    Blank cells are dropped, not coerced — an unlabelled item is missing data
    rather than a label, and alpha handles that.
    """
    per_annotator, items = {}, {}
    for path in paths:
        name = Path(path).stem
        table = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rid = r.get("row_id", "").strip()
                if not rid:
                    continue
                items.setdefault(rid, {
                    "context": r.get("context", ""),
                    "utterance": r.get("utterance", ""),
                })
                value = (r.get(field) or "").strip()
                if value.lower() in MISSING:
                    continue
                table[rid] = value
                items[rid][f"{name}_{field}"] = value
                items[rid][f"{name}_confidence"] = r.get("confidence_1_to_5", "")
                items[rid][f"{name}_notes"] = r.get("notes", "")
        per_annotator[name] = table
    return per_annotator, items


def percent_agreement(a, b):
    shared = set(a) & set(b)
    if not shared:
        return float("nan"), 0
    hits = sum(1 for k in shared if a[k] == b[k])
    return hits / len(shared), len(shared)


def cohens_kappa(a, b):
    """Chance-corrected agreement for exactly two annotators."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan")
    n = len(shared)
    observed = sum(1 for k in shared if a[k] == b[k]) / n

    labels = {a[k] for k in shared} | {b[k] for k in shared}
    expected = sum(
        (sum(1 for k in shared if a[k] == lab) / n) *
        (sum(1 for k in shared if b[k] == lab) / n)
        for lab in labels
    )
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


def fleiss_kappa(per_annotator):
    """Chance-corrected agreement for 3+ annotators, complete items only."""
    ids = set.intersection(*(set(t) for t in per_annotator.values()))
    ids = sorted(ids)
    n_raters = len(per_annotator)
    if not ids or n_raters < 2:
        return float("nan"), 0

    labels = sorted({t[i] for t in per_annotator.values() for i in ids})
    counts = [[sum(1 for t in per_annotator.values() if t[i] == lab)
               for lab in labels] for i in ids]

    # P_i: proportion of rater pairs on item i that agreed
    p_items = [
        (sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in counts
    ]
    p_bar = sum(p_items) / len(ids)
    # p_j: overall proportion of assignments to label j
    p_labels = [sum(row[j] for row in counts) / (len(ids) * n_raters)
                for j in range(len(labels))]
    pe = sum(p * p for p in p_labels)
    if pe == 1:
        return float("nan"), len(ids)
    return (p_bar - pe) / (1 - pe), len(ids)


def krippendorff_alpha(per_annotator):
    """Nominal-scale alpha. Handles missing values and any number of raters.

    Computed from the coincidence matrix directly (Krippendorff 2004), so items
    labelled by only one annotator drop out on their own rather than needing to
    be filtered first.
    """
    by_item = defaultdict(list)
    for table in per_annotator.values():
        for rid, label in table.items():
            by_item[rid].append(label)
    usable = {i: v for i, v in by_item.items() if len(v) >= 2}
    if not usable:
        return float("nan"), 0

    coincidence = Counter()
    n_total = 0.0
    for values in usable.values():
        m = len(values)
        for x, y in itertools.permutations(values, 2):
            coincidence[(x, y)] += 1.0 / (m - 1)
        n_total += m

    labels = sorted({v for values in usable.values() for v in values})
    observed = sum(coincidence[(lab, lab)] for lab in labels)
    marginals = {lab: sum(coincidence[(lab, other)] for other in labels)
                 for lab in labels}

    expected = sum(marginals[lab] * (marginals[lab] - 1) for lab in labels) / (n_total - 1)
    do = n_total - observed          # observed disagreement
    de = n_total - expected          # expected disagreement
    if de == 0:
        return float("nan"), len(usable)
    return 1 - do / de, len(usable)


def per_label_agreement(per_annotator, label_space):
    """For each label: how often did annotators who used it agree with each other?

    A decent overall kappa can still hide one category nobody applies
    consistently.
    """
    names = sorted(per_annotator)
    stats = {}
    for lab in label_space:
        agree = total = 0
        for a, b in itertools.combinations(names, 2):
            ta, tb = per_annotator[a], per_annotator[b]
            for rid in set(ta) & set(tb):
                if ta[rid] == lab or tb[rid] == lab:
                    total += 1
                    agree += ta[rid] == tb[rid]
        if total:
            stats[lab] = (agree / total, total)
    return stats


def confusion(per_annotator, label_space):
    """Pairwise confusion pooled over annotator pairs, as an unordered count."""
    names = sorted(per_annotator)
    table = Counter()
    for a, b in itertools.combinations(names, 2):
        ta, tb = per_annotator[a], per_annotator[b]
        for rid in set(ta) & set(tb):
            if ta[rid] != tb[rid]:
                table[tuple(sorted((ta[rid], tb[rid])))] += 1
    return table


def report_field(per_annotator, field, label_space):
    names = sorted(per_annotator)
    print(f"\n{'='*68}\n{field.upper()}\n{'='*68}")

    print("Pairwise:")
    for a, b in itertools.combinations(names, 2):
        pa, n = percent_agreement(per_annotator[a], per_annotator[b])
        k = cohens_kappa(per_annotator[a], per_annotator[b])
        print(f"  {a} vs {b:<16} n={n:<5} agreement={pa:.1%}  Cohen k={k:.3f}")

    if len(names) >= 3:
        k, n = fleiss_kappa(per_annotator)
        print(f"\nFleiss kappa ({len(names)} annotators, {n} complete items): {k:.3f}")

    alpha, n = krippendorff_alpha(per_annotator)
    print(f"Krippendorff alpha ({n} items with 2+ labels): {alpha:.3f}")

    stats = per_label_agreement(per_annotator, label_space)
    if stats:
        print("\nPer-label agreement (of pairs where either annotator used it):")
        for lab, (rate, total) in sorted(stats.items(), key=lambda kv: kv[1][0]):
            flag = "  <-- weakest" if rate == min(v[0] for v in stats.values()) else ""
            print(f"  {lab:<16} {rate:>6.1%}  (n={total}){flag}")

    conf = confusion(per_annotator, label_space)
    if conf:
        print("\nMost common disagreements:")
        for (x, y), count in conf.most_common(8):
            print(f"  {x} vs {y:<16} {count}")

    return alpha


def write_adjudication(per_annotator_maxim, per_annotator_vtype, items, out_path):
    """Every item that disagreed, or that anyone flagged low-confidence."""
    names = sorted(per_annotator_maxim)
    rows = []
    for rid, item in items.items():
        maxims = {n: per_annotator_maxim[n].get(rid) for n in names}
        vtypes = {n: per_annotator_vtype[n].get(rid) for n in names}
        given = [v for v in maxims.values() if v]
        low_conf = any(
            (item.get(f"{n}_confidence") or "").strip() in {"1", "2"} for n in names
        )
        missing = len(given) < len(names)
        disagree = len(set(given)) > 1
        vdisagree = len({v for v in vtypes.values() if v}) > 1
        if not (disagree or vdisagree or low_conf or missing):
            continue

        reason = ("maxim disagreement" if disagree else
                  "violation_type disagreement" if vdisagree else
                  "low confidence" if low_conf else "incomplete")
        row = {"row_id": rid, "reason": reason,
               "context": item["context"], "utterance": item["utterance"]}
        for n in names:
            row[f"{n}_maxim"] = maxims[n] or ""
            row[f"{n}_violation_type"] = vtypes[n] or ""
            row[f"{n}_confidence"] = item.get(f"{n}_confidence", "")
            row[f"{n}_notes"] = item.get(f"{n}_notes", "")
        row["adjudicated_maxim"] = ""
        row["adjudicated_violation_type"] = ""
        row["adjudication_reason"] = ""
        rows.append(row)

    if not rows:
        print("\nNo items need adjudication.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    breakdown = Counter(r["reason"] for r in rows)
    print(f"\nAdjudication sheet: {out_path}")
    print(f"  {len(rows)} items need a decision — {dict(breakdown)}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", nargs="+", help="Filled-in annotation CSVs, one per annotator")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Adjudication sheet path")
    args = ap.parse_args()

    paths = [Path(p) for p in args.annotations]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"No such file: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 1
    if len(paths) < 2:
        print("Need at least two annotation files to measure agreement.", file=sys.stderr)
        return 1

    maxim_tables, items = load_annotations(paths, "maxim")
    vtype_tables, _ = load_annotations(paths, "violation_type")

    print(f"Annotators: {', '.join(sorted(maxim_tables))}")
    print(f"Items in sheet: {len(items)}")
    for name in sorted(maxim_tables):
        labelled = len(maxim_tables[name])
        print(f"  {name:<16} labelled {labelled}/{len(items)}"
              f"   {dict(Counter(maxim_tables[name].values()).most_common())}")

    alpha_maxim = report_field(maxim_tables, "maxim", MAXIMS + EXTRA_MAXIM_LABELS)
    report_field(vtype_tables, "violation_type", VIOLATION_TYPES)

    n_adj = write_adjudication(maxim_tables, vtype_tables, items, Path(args.out))

    print(f"\n{'='*68}")
    if alpha_maxim == alpha_maxim and alpha_maxim < 0.40:
        print("Maxim alpha is below 0.40. The label scheme is not yet reliable —")
        print("fix the guidelines and re-annotate before training anything on it.")
    elif alpha_maxim == alpha_maxim and alpha_maxim < 0.60:
        print("Maxim alpha is moderate. Usable, but report it alongside every")
        print("model number: the classifier's ceiling is the annotators' agreement.")
    else:
        print("Maxim alpha is substantial. Report it with the model results.")
    print(f"Next: adjudicate the {n_adj} flagged items, then fold the resolved")
    print("labels back into data/test/test_natural.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
