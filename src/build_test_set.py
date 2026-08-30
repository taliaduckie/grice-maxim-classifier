"""Build the frozen held-out test set.

    test_natural.csv          50 human-annotated Reddit pairs
    test_synthetic.csv        stratified 20% of the 367 hand-written pairs
    test_natural_pending.csv  100 more natural pairs, labels stripped

Anything appearing in those is removed from the training corpus and written to
corpus_train.csv. 43 of the 50 gold pairs were already in corpus.csv, so scores
previously computed on them were training scores.

manifest.json pins each split by row-id hash so later drift is detectable.

Usage:
    python3 src/build_test_set.py --gold ~/Downloads/gold_annotated.csv
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


from provenance import (
    SCRAPE_COLUMNS, annotate, key, norm, read_csv_tolerant, row_id,
)
from paths import CORPUS_PATH, DATA_DIR, TEST_DIR, TRAIN_PATH, rel
from labels import (
    MAXIMS, VIOLATION_TYPES,
    normalize_violation_type as normalize_vtype,
)

# Held-out fraction of the synthetic pairs. 20% of 367 is 73 rows, ~15 per
# class — small, but the synthetic set is balanced so every class is present.
SYNTHETIC_TEST_FRACTION = 0.20

# How many further natural pairs to queue up for independent annotation.
PENDING_TEST_SIZE = 100

SEED = 20260809

def load_gold(path: Path):
    """Read the human-annotated gold file, repairing broken rows.

    Returns (clean_rows, rejected). Rows whose gold_maxim isn't one of the five
    labels are rejected, not guessed at: a shifted column would otherwise become
    a silent relabel.
    """
    rows, repaired, dropped = read_csv_tolerant(path, SCRAPE_COLUMNS)
    print(f"  read {len(rows)} rows ({repaired} repaired, {dropped} unrecoverable)")

    clean, rejected = [], []
    for r in rows:
        maxim = norm(r.get("gold_maxim"))
        vtype = normalize_vtype(r.get("gold_violation_type"))
        if maxim not in MAXIMS or vtype not in VIOLATION_TYPES:
            rejected.append({**r, "_reason": f"maxim={maxim!r} vtype={vtype!r}"})
            continue
        clean.append({
            "utterance": norm(r["utterance"]),
            "context": norm(r["context"]),
            "maxim": maxim,
            "violation_type": vtype,
            "source": "natural",
            "subreddit": norm(r.get("subreddit")),
            "post_title": norm(r.get("post_title")),
            "row_id": row_id(r),
        })
    return clean, rejected


def stratified_sample(rows, fraction=None, size=None, seed=SEED, by="maxim"):
    """Sample proportionally within each class, with a fixed seed."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[by]].append(r)

    picked = []
    for label in sorted(buckets):
        bucket = sorted(buckets[label], key=lambda r: r["row_id"])
        rng.shuffle(bucket)
        if fraction is not None:
            n = max(1, round(len(bucket) * fraction))
        else:
            n = max(1, round(size * len(bucket) / len(rows)))
        picked.extend(bucket[:n])
    return picked


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dist(rows, field="maxim"):
    return dict(Counter(r[field] for r in rows).most_common())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="Path to gold_annotated.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utterance", "context", "maxim", "violation_type",
                  "source", "subreddit", "post_title", "row_id"]

    # ---- natural test core -------------------------------------------------
    print("Loading gold annotations...")
    gold, rejected = load_gold(Path(args.gold).expanduser())
    print(f"  {len(gold)} usable, {len(rejected)} rejected as malformed")
    for r in rejected:
        print(f"    rejected: {r['_reason']}  {r['utterance'][:60]!r}")

    seen = set()
    natural_test = []
    for r in gold:
        if r["row_id"] in seen:
            continue
        seen.add(r["row_id"])
        natural_test.append(r)
    if len(natural_test) < len(gold):
        print(f"  dropped {len(gold) - len(natural_test)} duplicate rows")

    # ---- synthetic test split ---------------------------------------------
    corpus = annotate(CORPUS_PATH)
    synthetic = [r for r in corpus if r["source"] == "synthetic"]
    synthetic_test = stratified_sample(synthetic, fraction=SYNTHETIC_TEST_FRACTION,
                                       seed=args.seed)

    # ---- pending natural pool ---------------------------------------------
    # Sampled from natural rows not already spoken for. Labels are stripped
    # rather than shown: this project has already been bitten once by
    # annotators anchoring on a pre-filled value (CHANGELOG, ed9a122).
    test_ids = {r["row_id"] for r in natural_test} | {r["row_id"] for r in synthetic_test}
    natural_pool = [r for r in corpus
                    if r["source"] == "natural" and r["row_id"] not in test_ids]
    pending = stratified_sample(natural_pool, size=PENDING_TEST_SIZE, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(pending)
    pending_sheet = [{
        "row_id": r["row_id"],
        "context": r["context"],
        "utterance": r["utterance"],
        "maxim": "",
        "violation_type": "",
        "confidence_1_to_5": "",
        "notes": "",
        "subreddit": r["subreddit"],
    } for r in pending]

    # ---- training corpus, with every test row removed ----------------------
    held_out = test_ids | {r["row_id"] for r in pending}
    # a finalized v2 natural set is matched by text: its items are re-paired
    # (question, comment), so the same comment sits under a different row_id
    # here as a context or utterance
    v2_path = TEST_DIR / "test_natural_v2.csv"
    v2_texts = set()
    if v2_path.exists():
        with open(v2_path, newline="", encoding="utf-8") as f:
            v2_texts = {norm(r["utterance"]).casefold() for r in csv.DictReader(f)}
    train_rows = [r for r in corpus if r["row_id"] not in held_out
                  and norm(r["utterance"]).casefold() not in v2_texts
                  and norm(r["context"]).casefold() not in v2_texts]
    removed = len(corpus) - len(train_rows)
    if v2_texts:
        print(f"  (test_natural_v2 present: its texts are excluded from training too)")

    # ---- write -------------------------------------------------------------
    paths = {
        "test_natural": TEST_DIR / "test_natural.csv",
        "test_synthetic": TEST_DIR / "test_synthetic.csv",
        "test_natural_pending": TEST_DIR / "test_natural_pending.csv",
        "corpus_train": TRAIN_PATH,
    }
    write_csv(paths["test_natural"], natural_test, fieldnames)
    write_csv(paths["test_synthetic"], synthetic_test, fieldnames)
    write_csv(paths["test_natural_pending"], pending_sheet, list(pending_sheet[0]))
    write_csv(paths["corpus_train"], train_rows, fieldnames)

    manifest = {
        "seed": args.seed,
        "source_corpus": {"path": str(CORPUS_PATH.relative_to(DATA_DIR.parent)),
                          "rows": len(corpus), "sha256_16": file_hash(CORPUS_PATH)},
        "splits": {},
    }
    for name, path in paths.items():
        rows = {"test_natural": natural_test, "test_synthetic": synthetic_test,
                "test_natural_pending": pending, "corpus_train": train_rows}[name]
        manifest["splits"][name] = {
            "path": str(path.relative_to(DATA_DIR.parent)),
            "rows": len(rows),
            "sha256_16": file_hash(path),
            "row_ids": sorted(r["row_id"] for r in rows),
            "maxim_distribution": dist(rows) if name != "test_natural_pending" else None,
        }
    (TEST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # ---- report ------------------------------------------------------------
    print(f"\n{'='*66}\nFROZEN TEST SET\n{'='*66}")
    print(f"test_natural          {len(natural_test):>5} rows   {dist(natural_test)}")
    print(f"test_synthetic        {len(synthetic_test):>5} rows   {dist(synthetic_test)}")
    print(f"test_natural_pending  {len(pending):>5} rows   (labels stripped for annotation)")
    print(f"corpus_train          {len(train_rows):>5} rows   "
          f"({removed} removed as held-out)")
    print(f"  train by source: {dist(train_rows, 'source')}")
    print(f"  train by maxim:  {dist(train_rows)}")

    contaminated = sum(1 for r in corpus if r["row_id"] in
                       {g["row_id"] for g in natural_test})
    print(f"\nContamination found and removed: {contaminated} of the "
          f"{len(natural_test)} gold rows were in the training corpus.")

    missing = [m for m in MAXIMS if m not in dist(natural_test)]
    if missing:
        print(f"\n!! test_natural has NO examples of: {', '.join(missing)}.")
        print("   A test set missing Cooperative cannot measure the "
              "register->Cooperative shortcut,")
        print("   which is the failure mode this project already knows it has. "
              "Annotating")
        print("   test_natural_pending.csv fixes this — it is sampled across all "
              "five classes.")

    print(f"\nManifest: {(TEST_DIR / 'manifest.json').relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
