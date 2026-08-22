"""Tag each corpus row as synthetic or natural and recover its subreddit.

corpus.csv has no provenance column, so the synthetic-vs-natural ablations can't
be expressed. The 367 synthetic pairs are byte-identical in corpus_367.csv;
everything else is Reddit, and the subreddit joins back from data/raw/*.csv on
(utterance, context).

build_test_set.py and ablations.py import the join from here so it stays
consistent.
"""

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CORPUS_PATH = DATA_DIR / "annotated" / "corpus.csv"
SYNTHETIC_PATH = DATA_DIR / "annotated" / "corpus_367.csv"
RAW_DIR = DATA_DIR / "raw"

# The nine columns the scrapers write. Rows with more fields than this are
# almost always a context that contained an unescaped comma — see repair_row.
SCRAPE_COLUMNS = [
    "utterance", "context", "predicted_maxim", "predicted_violation_type",
    "confidence", "gold_maxim", "gold_violation_type", "subreddit", "post_title",
]

_WS = re.compile(r"\s+")


def norm(text) -> str:
    """Whitespace- and quote-insensitive form used for all cross-file joins.

    The same text round-trips through several CSV writers with different
    quoting, so it can pick up stray outer quotes or have newlines collapsed.
    """
    if text is None:
        return ""
    return _WS.sub(" ", str(text)).strip().strip('"').strip()


def key(row) -> tuple:
    """Join key for a (utterance, context) pair."""
    return (norm(row.get("utterance")), norm(row.get("context")))


def row_id(row) -> str:
    """Stable 12-hex-char id for a pair. Used to freeze the test set by hash."""
    u, c = key(row)
    return hashlib.sha256(f"{u}\x00{c}".encode()).hexdigest()[:12]


def repair_row(fields, n_expected, context_index=1):
    """Rejoin fields that a broken CSV writer split mid-context.

    Some rows were written with quoting that didn't survive a comma inside the
    context. The trailing columns are always last and the utterance always
    first, so anything in between belongs to the context.

    Returns None for short rows — that damage isn't recoverable.
    """
    if len(fields) == n_expected:
        return fields
    if len(fields) < n_expected:
        return None
    n_trailing = n_expected - context_index - 1
    context = ",".join(fields[context_index:len(fields) - n_trailing])
    return fields[:context_index] + [context] + fields[len(fields) - n_trailing:]


def read_csv_tolerant(path, expected_columns=None):
    """Read a CSV, repairing over-split rows instead of dying on them.

    Returns (rows, n_repaired, n_dropped). pandas.read_csv raises a ParserError
    on the first bad row and loses the rest of the file with it.
    """
    with open(path, newline="", encoding="utf-8") as f:
        records = list(csv.reader(f))
    if not records:
        return [], 0, 0

    header = records[0]
    columns = expected_columns or header
    n = len(columns)
    context_index = columns.index("context") if "context" in columns else 1

    rows, repaired, dropped = [], 0, 0
    for fields in records[1:]:
        if len(fields) != n:
            fixed = repair_row(fields, n, context_index)
            if fixed is None:
                dropped += 1
                continue
            repaired += 1
            fields = fixed
        rows.append(dict(zip(columns, fields)))
    return rows, repaired, dropped


def synthetic_keys() -> set:
    """Keys of the 367 hand-written pairs."""
    with open(SYNTHETIC_PATH, newline="", encoding="utf-8") as f:
        return {key(r) for r in csv.DictReader(f)}


def subreddit_index() -> dict:
    """Map every scraped (utterance, context) pair to its subreddit and title."""
    index = {}
    for path in sorted(RAW_DIR.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        expected = SCRAPE_COLUMNS if len(header) == len(SCRAPE_COLUMNS) else header
        rows, _, _ = read_csv_tolerant(path, expected)
        for r in rows:
            k = key(r)
            if k not in index:
                index[k] = {
                    "subreddit": norm(r.get("subreddit")),
                    "post_title": norm(r.get("post_title")),
                    "scrape_file": path.name,
                }
    return index


def annotate(corpus_path=CORPUS_PATH) -> list:
    """Return corpus rows with source / subreddit / post_title / row_id added."""
    synth = synthetic_keys()
    subs = subreddit_index()

    with open(corpus_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        k = key(r)
        is_synth = k in synth
        meta = subs.get(k, {})
        out.append({
            "utterance": r["utterance"],
            "context": r["context"],
            "maxim": r["maxim"],
            "violation_type": r["violation_type"],
            "source": "synthetic" if is_synth else "natural",
            "subreddit": "" if is_synth else meta.get("subreddit", ""),
            "post_title": "" if is_synth else meta.get("post_title", ""),
            "row_id": row_id(r),
        })
    return out


def main():
    out_path = DATA_DIR / "annotated" / "corpus_provenance.csv"
    rows = annotate()

    fieldnames = ["utterance", "context", "maxim", "violation_type",
                  "source", "subreddit", "post_title", "row_id"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    src_counts = Counter(r["source"] for r in rows)
    print(f"Wrote {len(rows)} rows to {out_path.relative_to(DATA_DIR.parent)}")
    print(f"\nProvenance: {dict(src_counts)}")

    unmapped = sum(1 for r in rows if r["source"] == "natural" and not r["subreddit"])
    print(f"Natural rows with no subreddit recovered: {unmapped}"
          f" / {src_counts['natural']}")

    print("\nLabel distribution by source:")
    header = ["maxim", "synthetic", "natural", "natural %"]
    print(f"  {header[0]:<13}{header[1]:>10}{header[2]:>10}{header[3]:>11}")
    for maxim in ["Cooperative", "Quantity", "Quality", "Relation", "Manner"]:
        s = sum(1 for r in rows if r["source"] == "synthetic" and r["maxim"] == maxim)
        n = sum(1 for r in rows if r["source"] == "natural" and r["maxim"] == maxim)
        pct = n / src_counts["natural"] if src_counts["natural"] else 0
        print(f"  {maxim:<13}{s:>10}{n:>10}{pct:>10.1%}")

    print("\nTop subreddits:")
    for sub, count in Counter(
        r["subreddit"] for r in rows if r["subreddit"]
    ).most_common(10):
        print(f"  {sub:<20}{count:>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
