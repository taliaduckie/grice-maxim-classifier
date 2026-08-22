"""Rebuild the natural pairs as question->answer adjacency pairs.

The scraper paired top-level comment with reply and dropped the post title,
which on AskReddit-style subs is the only actual question in the exchange. Only
4% of test_natural contexts contain a question mark against 88% of the synthetic
ones. See results/foundation_report.md §11.

No re-scrape needed: every row in data/raw/*.csv is a depth-1 pair, so the
context column is already a top-level comment. Re-pairing (post_title, context)
recovers the adjacency pair.

Labels come out blank — the old (comment, reply) label says nothing about
(question, comment).

Usage:
    python3 src/rebuild_pairs.py [--per-thread 5] [--allow-non-questions]
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from provenance import (
    DATA_DIR, RAW_DIR, SCRAPE_COLUMNS, norm, read_csv_tolerant, row_id,
)

OUT_DIR = DATA_DIR / "annotated"

# Cap per thread. test_natural drew 50 items from only 4 threads, so its
# effective sample size was well below 50 — items in a thread share topic and
# register.
PER_THREAD_DEFAULT = 6

MIN_CHARS = 15
MAX_CHARS = 900


def load_raw():
    """Every scraped row that carries a post title."""
    rows = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        expected = SCRAPE_COLUMNS if len(header) == len(SCRAPE_COLUMNS) else header
        got, _, _ = read_csv_tolerant(path, expected)
        for r in got:
            if norm(r.get("post_title")):
                r["_file"] = path.name
                rows.append(r)
    return rows


def verify_depth_one(rows):
    """Confirm no context is itself someone's reply.

    If this fails, some contexts are mid-thread comments rather than top-level
    ones, and pairing them with the post title would invent an adjacency that
    never existed.
    """
    contexts = {norm(r["context"]) for r in rows}
    utterances = {norm(r["utterance"]) for r in rows}
    overlap = contexts & utterances
    return len(overlap), sorted(overlap)[:3]


def build(rows, per_thread, require_question):
    """One row per distinct top-level comment, capped per thread."""
    by_thread = defaultdict(list)
    seen = set()
    skipped = Counter()

    for r in rows:
        question = norm(r["post_title"])
        answer = norm(r["context"])

        if require_question and not question.endswith("?"):
            skipped["title is not a question"] += 1
            continue
        if not (MIN_CHARS <= len(answer) <= MAX_CHARS):
            skipped["answer too short or too long"] += 1
            continue

        key = (question, answer)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)

        by_thread[question].append({
            "row_id": row_id({"utterance": answer, "context": question}),
            "context": question,
            "utterance": answer,
            "maxim": "",
            "violation_type": "",
            "confidence_1_to_5": "",
            "notes": "",
            "subreddit": norm(r.get("subreddit")),
            "thread": question,
            "source": "natural",
            "pairing": "title_to_toplevel",
            "scrape_file": r["_file"],
        })

    out = []
    for question in sorted(by_thread):
        bucket = sorted(by_thread[question], key=lambda x: x["row_id"])
        if len(bucket) > per_thread:
            skipped["over per-thread cap"] += len(bucket) - per_thread
        out.extend(bucket[:per_thread])
    return out, skipped


def contamination(pairs):
    """Flag any answer text that already appears in the training corpus.

    The same comment can be a `context` under the old pairing and an
    `utterance` here. Not a duplicate row, but a model trained on the old corpus
    has read the text, so pull these from training before using them as test data.
    """
    train_path = OUT_DIR / "corpus_train.csv"
    if not train_path.exists():
        return set()
    texts = set()
    with open(train_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            texts.add(norm(r["utterance"]))
            texts.add(norm(r["context"]))
    return {p["row_id"] for p in pairs if p["utterance"] in texts}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-thread", type=int, default=PER_THREAD_DEFAULT)
    ap.add_argument("--require-question", action="store_true", default=True)
    ap.add_argument("--allow-non-questions", dest="require_question",
                    action="store_false",
                    help="Keep threads whose title is not interrogative "
                         "(AITA, tifu). Their titles are still the discourse "
                         "purpose, but the maxims are harder to apply.")
    ap.add_argument("--out", default=str(OUT_DIR / "natural_qa_pairs.csv"))
    args = ap.parse_args()

    rows = load_raw()
    print(f"Raw rows carrying a post title: {len(rows)}")

    n_overlap, examples = verify_depth_one(rows)
    if n_overlap:
        print(f"\n!! {n_overlap} contexts also appear as replies, so they are "
              "not all top-level.")
        print("   Pairing those with the post title would invent an adjacency "
              "that never existed.")
        for e in examples:
            print(f"     {e[:70]!r}")
        print("   Refusing to build. Re-scrape with --pairing title_to_toplevel "
              "instead.")
        return 1
    print("Verified: no context appears as a reply, so every context is "
          "top-level.")

    pairs, skipped = build(rows, args.per_thread, args.require_question)
    if not pairs:
        print("No pairs survived filtering.", file=sys.stderr)
        return 1

    dirty = contamination(pairs)
    for p in pairs:
        p["seen_in_training"] = "yes" if p["row_id"] in dirty else "no"

    out_path = Path(args.out)
    fieldnames = ["row_id", "context", "utterance", "maxim", "violation_type",
                  "confidence_1_to_5", "notes", "subreddit", "source",
                  "pairing", "seen_in_training", "scrape_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs)

    threads = Counter(p["thread"] for p in pairs)
    subs = Counter(p["subreddit"] for p in pairs)
    print(f"\n{'='*66}\nREBUILT PAIRS\n{'='*66}")
    print(f"Pairs written      : {len(pairs)}")
    print(f"Distinct threads   : {len(threads)}  "
          f"(test_natural had 4)")
    print(f"Distinct subreddits: {len(subs)}")
    print(f"Median per thread  : {sorted(threads.values())[len(threads)//2]}")
    print(f"Already seen in training: {len(dirty)} "
          f"— pull these from corpus_train.csv before using any as test data")

    if skipped:
        print("\nSkipped:")
        for reason, n in skipped.most_common():
            print(f"  {reason:<28}{n:>5}")

    print("\nTop subreddits:")
    for s, n in subs.most_common(8):
        print(f"  {s or '(unknown)':<24}{n:>4}")

    q = sum(1 for p in pairs if p["context"].endswith("?"))
    print(f"\nContexts that are questions: {q}/{len(pairs)} "
          f"({q/len(pairs):.0%})  — test_natural was 4%")
    print(f"\nWrote {out_path.relative_to(DATA_DIR.parent)}")
    print("Labels are blank by design. Annotate under "
          "docs/annotation_guidelines.md;")
    print("the old (comment, reply) label does not transfer to "
          "(question, comment).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
