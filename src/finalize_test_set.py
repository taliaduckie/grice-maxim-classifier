"""Turn finished annotations into a frozen natural test set.

Inputs:
    data/test/annotations/<name>.csv   one per annotator, as annotate.py writes
    data/test/adjudication_sheet.csv   from agreement.py, with the
                                       adjudicated_* columns filled in (optional)
    data/derived/natural_qa_pairs.csv  the sheet that was annotated; supplies
                                       subreddit / thread / seen_in_training

Label resolution per item, in order:
    1. an adjudicated label, if the sheet has one
    2. the unanimous label, if every annotator who saw it agrees
    3. the only label, if just one annotator saw it (flagged single-annotator)
    4. otherwise unresolved — written to data/test/unresolved.csv, not included

Outputs:
    data/test/test_natural_v2.csv      same columns as test_natural.csv, plus
                                       thread / label_source / n_annotators
    data/test/manifest.json            gains a test_natural_v2 split
    data/derived/corpus_train.csv      rows whose text appears in v2 removed

Usage:
    python3 src/finalize_test_set.py
    python3 src/finalize_test_set.py --min-annotators 2
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from labels import MAXIMS, VIOLATION_TYPES, normalize_violation_type
from paths import (
    DATA_DIR, MANIFEST_PATH, QA_PAIRS_PATH, TEST_DIR, TRAIN_PATH, rel,
)
from provenance import norm

ANNOTATIONS_DIR = TEST_DIR / "annotations"
ADJUDICATION_PATH = TEST_DIR / "adjudication_sheet.csv"
OUT_PATH = TEST_DIR / "test_natural_v2.csv"
UNRESOLVED_PATH = TEST_DIR / "unresolved.csv"

OUT_COLUMNS = ["utterance", "context", "maxim", "violation_type", "source",
               "subreddit", "post_title", "row_id", "thread", "label_source",
               "n_annotators"]

MISSING = {"", "-", "na", "n/a", "?"}


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_annotations(paths):
    """{row_id: [(annotator, maxim, vtype), ...]} — blanks dropped."""
    votes = defaultdict(list)
    for path in paths:
        name = Path(path).stem
        for r in read(path):
            rid = (r.get("row_id") or "").strip()
            maxim = (r.get("maxim") or "").strip()
            if not rid or maxim.lower() in MISSING:
                continue
            vtype = normalize_violation_type(r.get("violation_type"))
            votes[rid].append((name, maxim, vtype))
    return votes


def load_adjudications(path):
    """{row_id: (maxim, vtype)} for rows where a decision was recorded."""
    if not path.exists():
        return {}
    out = {}
    for r in read(path):
        maxim = (r.get("adjudicated_maxim") or "").strip()
        if maxim.lower() in MISSING:
            continue
        vtype = normalize_violation_type(r.get("adjudicated_violation_type"))
        out[r["row_id"].strip()] = (maxim, vtype)
    return out


def resolve(rid, votes, adjudicated, min_annotators):
    """Return (maxim, vtype, source, n_annotators) or None if unresolved."""
    n = len(votes)
    if rid in adjudicated:
        maxim, vtype = adjudicated[rid]
        return maxim, vtype, "adjudicated", n
    if n < min_annotators:
        return None
    maxims = {m for _, m, _ in votes}
    if len(maxims) == 1:
        maxim = maxims.pop()
        vtypes = Counter(v for _, _, v in votes)
        # violation_type is the harder call; take the majority, "unknown" on a tie
        top = vtypes.most_common()
        vtype = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "unknown"
        source = "unanimous" if n > 1 else "single-annotator"
        return maxim, vtype, source, n
    return None


def validate(maxim, vtype):
    if maxim not in MAXIMS:
        return f"maxim {maxim!r} not in schema"
    if vtype not in VIOLATION_TYPES:
        return f"violation_type {vtype!r} not in schema"
    if maxim == "Cooperative" and vtype != "none":
        return "Cooperative must have violation_type=none"
    if maxim != "Cooperative" and vtype == "none":
        return f"{maxim} cannot have violation_type=none"
    return None


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pull_from_training(test_rows, train_path):
    """Remove training rows whose utterance or context text is a v2 utterance.

    Returns (kept_rows, n_removed). The same comment can be a `context` under
    the old comment->reply pairing, so matching is on text, not row_id.
    Case- and whitespace-insensitive: over-removing is the safe direction.
    """
    fold = lambda t: norm(t).casefold()
    texts = {fold(r["utterance"]) for r in test_rows}
    train = read(train_path)
    kept = [r for r in train
            if fold(r["utterance"]) not in texts and fold(r["context"]) not in texts]
    return kept, len(train) - len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="*",
                    default=sorted(str(p) for p in ANNOTATIONS_DIR.glob("*.csv")))
    ap.add_argument("--adjudication", default=str(ADJUDICATION_PATH))
    ap.add_argument("--sheet", default=str(QA_PAIRS_PATH))
    ap.add_argument("--min-annotators", type=int, default=1,
                    help="Items seen by fewer annotators are left unresolved.")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--no-train-pull", action="store_true",
                    help="Do not rewrite corpus_train.csv.")
    args = ap.parse_args()

    if not args.annotations:
        print(f"No annotation files in {rel(ANNOTATIONS_DIR)}.", file=sys.stderr)
        return 1

    sheet = {r["row_id"]: r for r in read(args.sheet)}
    votes = load_annotations(args.annotations)
    adjudicated = load_adjudications(Path(args.adjudication))
    print(f"Annotators : {', '.join(Path(p).stem for p in args.annotations)}")
    print(f"Sheet items: {len(sheet)}   annotated: {len(votes)}   "
          f"adjudicated: {len(adjudicated)}")

    resolved, unresolved, invalid = [], [], []
    for rid, item in sheet.items():
        result = resolve(rid, votes.get(rid, []), adjudicated, args.min_annotators)
        if result is None:
            if votes.get(rid):
                unresolved.append({
                    "row_id": rid, "context": item["context"],
                    "utterance": item["utterance"],
                    "labels": "; ".join(f"{a}={m}/{v}" for a, m, v in votes[rid]),
                })
            continue
        maxim, vtype, source, n = result
        problem = validate(maxim, vtype)
        if problem:
            invalid.append((rid, problem))
            continue
        resolved.append({
            "utterance": item["utterance"],
            "context": item["context"],
            "maxim": maxim,
            "violation_type": vtype,
            "source": "natural",
            "subreddit": item.get("subreddit", ""),
            "post_title": item["context"],
            "row_id": rid,
            "thread": item.get("thread") or item["context"],
            "label_source": source,
            "n_annotators": n,
        })

    if not resolved:
        print("Nothing resolved.", file=sys.stderr)
        return 1

    out = Path(args.out)
    write_csv(out, resolved, OUT_COLUMNS)
    unresolved_path = out.parent / UNRESOLVED_PATH.name
    if unresolved:
        write_csv(unresolved_path, unresolved, list(unresolved[0]))

    # ---- training pull ----------------------------------------------------
    n_pulled = 0
    if not args.no_train_pull and TRAIN_PATH.exists():
        kept, n_pulled = pull_from_training(resolved, TRAIN_PATH)
        if n_pulled:
            write_csv(TRAIN_PATH, kept, list(kept[0]))

    # ---- manifest ---------------------------------------------------------
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"splits": {}}
    manifest["splits"]["test_natural_v2"] = {
        "path": rel(out),
        "rows": len(resolved),
        "sha256_16": file_hash(out),
        "row_ids": sorted(r["row_id"] for r in resolved),
        "maxim_distribution": dict(Counter(r["maxim"] for r in resolved).most_common()),
        "label_source": dict(Counter(r["label_source"] for r in resolved)),
        "annotators": [Path(p).stem for p in args.annotations],
    }
    if TRAIN_PATH.exists():
        train_rows = read(TRAIN_PATH)
        manifest["splits"]["corpus_train"] = {
            "path": rel(TRAIN_PATH), "rows": len(train_rows),
            "sha256_16": file_hash(TRAIN_PATH),
            "row_ids": sorted(r["row_id"] for r in train_rows),
            "maxim_distribution": dict(Counter(r["maxim"] for r in train_rows).most_common()),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # ---- report -----------------------------------------------------------
    dist = Counter(r["maxim"] for r in resolved)
    src = Counter(r["label_source"] for r in resolved)
    threads = len({r["thread"] for r in resolved})
    print(f"\n{'='*64}\nTEST_NATURAL_V2\n{'='*64}")
    print(f"Resolved   : {len(resolved)} / {len(sheet)}   "
          f"({dict(src)})")
    print(f"Unresolved : {len(unresolved)}"
          + (f"   -> {rel(unresolved_path)}" if unresolved else ""))
    if invalid:
        print(f"Invalid    : {len(invalid)}")
        for rid, why in invalid[:5]:
            print(f"    {rid}: {why}")
    print(f"Threads    : {threads}")
    print(f"Labels     : {dict(dist.most_common())}")
    coop = dist.get("Cooperative", 0) / len(resolved)
    print(f"Cooperative: {coop:.0%}  (old test_natural: 0%)")
    q = sum(1 for r in resolved if r["context"].endswith("?"))
    print(f"Questions  : {q}/{len(resolved)}  (old test_natural: 4%)")
    print(f"Pulled from corpus_train: {n_pulled}")
    print(f"\nWrote {rel(out)} and updated {rel(manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
