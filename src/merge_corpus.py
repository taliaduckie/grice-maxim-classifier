import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


from paths import CORPUS_PATH
FIELDNAMES = ["utterance", "context", "maxim", "violation_type"]
from labels import MAXIMS, normalize_violation_type as normalize_vtype

VALID_MAXIMS = set(MAXIMS)
VALID_VIOLATION_TYPES = {
    "none", "flouting", "violating", "failed_flout",
    "opting_out", "clash", "unknown",
}


def _is_valid(row):
    if not row["utterance"]:
        return False
    if row["maxim"] not in VALID_MAXIMS:
        return False
    if row["violation_type"] not in VALID_VIOLATION_TYPES:
        return False
    return True


def _load_corpus():
    rows, keys = [], set()
    if not CORPUS_PATH.exists():
        return rows, keys
    with open(CORPUS_PATH) as f:
        for r in csv.DictReader(f):
            row = {
                "utterance": r["utterance"].strip('"'),
                "context": r["context"].strip('"'),
                "maxim": r["maxim"].strip(),
                "violation_type": r["violation_type"].strip(),
            }
            rows.append(row)
            keys.add((row["utterance"], row["context"]))
    return rows, keys


def _write(rows):
    with open(CORPUS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def _summary(rows, added, dupes, skipped):
    mc = Counter(r["maxim"] for r in rows)
    vc = Counter(r["violation_type"] for r in rows)
    print(f"Added: {added}, Duplicates: {dupes}, Skipped: {skipped}")
    print(f"Total: {len(rows)}")
    print(f"Maxim: {dict(mc)}")
    print(f"Violation: {dict(vc)}")


def merge_pipe_data(pipe_string: str) -> int:
    """Each line: utterance|context|maxim|violation_type"""
    existing, keys = _load_corpus()

    added = 0
    dupes = 0
    skipped = 0

    for line in pipe_string.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.rsplit("|", 3)
        if len(parts) != 4:
            print(f"  skipping malformed line: {line[:60]}...")
            skipped += 1
            continue

        row = {
            "utterance": parts[0].strip(),
            "context": parts[1].strip(),
            "maxim": parts[2].strip(),
            "violation_type": normalize_vtype(parts[3]),
        }

        if not _is_valid(row):
            skipped += 1
            continue

        key = (row["utterance"], row["context"])
        if key in keys:
            dupes += 1
            continue

        existing.append(row)
        keys.add(key)
        added += 1

    _write(existing)
    _summary(existing, added, dupes, skipped)
    return added


def merge_annotated_csv(csv_path: str) -> int:
    """CSV needs utterance, context, maxim, violation_type (or gold_* variants)"""
    existing, keys = _load_corpus()
    added = 0
    dupes = 0
    skipped = 0

    with open(csv_path) as f:
        for r in csv.DictReader(f):
            maxim = r.get("maxim") or r.get("gold_maxim", "")
            vtype = r.get("violation_type") or r.get("gold_violation_type", "")

            row = {
                "utterance": r.get("utterance", "").strip('"').strip(),
                "context": r.get("context", "").strip('"').strip(),
                "maxim": maxim.strip(),
                "violation_type": normalize_vtype(vtype),
            }

            if not _is_valid(row):
                skipped += 1
                continue

            key = (row["utterance"], row["context"])
            if key in keys:
                dupes += 1
                continue

            existing.append(row)
            keys.add(key)
            added += 1

    _write(existing)
    _summary(existing, added, dupes, skipped)
    return added


def merge_with_scraped(scraped_path: str, annotation_pipe_string: str) -> int:
    """Match annotations to scraped CSV by row order"""
    existing, keys = _load_corpus()

    scraped = []
    with open(scraped_path) as f:
        for r in csv.DictReader(f):
            scraped.append(r)

    annotation_lines = [
        line for line in annotation_pipe_string.strip().split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    added = 0
    dupes = 0
    skipped = 0

    for i, line in enumerate(annotation_lines):
        if i >= len(scraped):
            print(f"  more annotations than scraped rows, stopping at {i}")
            break

        parts = line.rsplit("|", 2)
        if len(parts) != 3:
            skipped += 1
            continue

        # use full utterance from scraped (parts[0] is just the prefix)
        row = {
            "utterance": scraped[i]["utterance"].strip('"'),
            "context": scraped[i]["context"].strip('"'),
            "maxim": parts[1].strip(),
            "violation_type": normalize_vtype(parts[2]),
        }

        if not _is_valid(row):
            skipped += 1
            continue

        key = (row["utterance"], row["context"])
        if key in keys:
            dupes += 1
            continue

        existing.append(row)
        keys.add(key)
        added += 1

    _write(existing)
    _summary(existing, added, dupes, skipped)
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge annotations into corpus.csv")
    parser.add_argument("--csv", help="Path to fully-annotated CSV.")
    parser.add_argument("--scraped", help="Path to scraped CSV (use with --annotations).")
    parser.add_argument("--annotations", help="Path to annotation file (pipe-separated).")
    args = parser.parse_args()

    if args.csv:
        merge_annotated_csv(args.csv)
    elif args.scraped and args.annotations:
        with open(args.annotations) as f:
            merge_with_scraped(args.scraped, f.read())
    else:
        parser.error("provide either --csv or both --scraped and --annotations")
