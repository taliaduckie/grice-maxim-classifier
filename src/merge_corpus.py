import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

CORPUS_PATH = Path(__file__).parent.parent / "data" / "annotated" / "corpus.csv"
FIELDNAMES = ["utterance", "context", "maxim", "violation_type"]
VALID_MAXIMS = {"Cooperative", "Quantity", "Quality", "Relation", "Manner"}
VALID_VIOLATION_TYPES = {
    "none", "flouting", "violating", "failed_flout",
    "opting_out", "clash", "unknown",
}


def normalize_violation_type(vtype: str) -> str:
    vtype = vtype.strip().lower()
    aliases = {
        "flout": "flouting",
        "violate": "violating",
        "violation": "violating",
        "sincere": "none",
        "no violation": "none",
        "": "unknown",
    }
    return aliases.get(vtype, vtype)


def _load_corpus() -> tuple[list, set]:
    rows = []
    keys = set()
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


def _validate_row(row: dict) -> Optional[str]:
    if not row["utterance"]:
        return "empty utterance"
    if row["maxim"] not in VALID_MAXIMS:
        return f"invalid maxim: {row['maxim']}"
    if row["violation_type"] not in VALID_VIOLATION_TYPES:
        return f"invalid violation_type: {row['violation_type']}"
    return None


def _write_corpus(rows: list):
    with open(CORPUS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list, added: int, dupes: int, skipped: int):
    mc = Counter(r["maxim"] for r in rows)
    vc = Counter(r["violation_type"] for r in rows)
    print(f"Added: {added}, Duplicates: {dupes}, Skipped (invalid): {skipped}")
    print(f"Total: {len(rows)}")
    print(f"Maxim:     {dict(mc)}")
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
            "violation_type": normalize_violation_type(parts[3]),
        }

        err = _validate_row(row)
        if err:
            print(f"  skipping ({err}): {row['utterance'][:50]}...")
            skipped += 1
            continue

        key = (row["utterance"], row["context"])
        if key in keys:
            dupes += 1
            continue

        existing.append(row)
        keys.add(key)
        added += 1

    _write_corpus(existing)
    _print_summary(existing, added, dupes, skipped)
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
                "violation_type": normalize_violation_type(vtype),
            }

            err = _validate_row(row)
            if err:
                skipped += 1
                continue

            key = (row["utterance"], row["context"])
            if key in keys:
                dupes += 1
                continue

            existing.append(row)
            keys.add(key)
            added += 1

    _write_corpus(existing)
    _print_summary(existing, added, dupes, skipped)
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

        # parts[0] is the utterance prefix (for sanity check), but we use
        # the full utterance from the scraped data for safety
        row = {
            "utterance": scraped[i]["utterance"].strip('"'),
            "context": scraped[i]["context"].strip('"'),
            "maxim": parts[1].strip(),
            "violation_type": normalize_violation_type(parts[2]),
        }

        err = _validate_row(row)
        if err:
            skipped += 1
            continue

        key = (row["utterance"], row["context"])
        if key in keys:
            dupes += 1
            continue

        existing.append(row)
        keys.add(key)
        added += 1

    _write_corpus(existing)
    _print_summary(existing, added, dupes, skipped)
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
