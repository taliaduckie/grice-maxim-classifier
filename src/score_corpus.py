"""
coherence_score.py

Compute semantic coherence between context (Q) and utterance (A) using
sentence-transformers embeddings. The idea: if the response is semantically
distant from the question, it's either a deliberate non-sequitur (flouting)
or a genuine failure to stay on topic (violating / failed_flout). but which?

The key insight: flouting SHOULD be low-coherence (the whole point is that
the response is irrelevant) but the deflection is doing communicative work —
it's legible as a refusal, a topic change, a power move. violating is ALSO
low-coherence but the speaker genuinely lost the thread. the surface form
is identical. the difference is intent.

What this script actually measures:
    L = cosine similarity between Q and A embeddings (semantic coherence)
    O = 1 - L (semantic distance / "off-topic-ness")

What to look for in the output:
    - Relation examples should cluster into two groups by O score
    - High O + labeled "flouting" = probably correct (deliberate deflection)
    - Low O + labeled "flouting" = suspicious (maybe it's actually cooperative?)
    - High O + labeled "violating" = probably correct (genuinely off-topic)
    - Low O + labeled "violating" = suspicious (maybe they ARE answering?)

This doesn't give you ground truth. It gives you candidates for relabeling.
The decision is still human. but now it's an informed human instead of a
vibes-based human. which is an improvement.

Usage:
    python coherence_score.py
    python coherence_score.py --data ../data/annotated/corpus.csv
    python coherence_score.py --data ../data/annotated/corpus.csv --output scores.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def compute_coherence(csv_path: str, output_path: str = None):
    """
    Compute L (coherence) and O (off-topic-ness) for every example in the corpus.
    """
    df = pd.read_csv(csv_path)
    assert "utterance" in df.columns and "context" in df.columns, (
        "CSV must have 'utterance' and 'context' columns."
    )

    # all-MiniLM-L6-v2 is small, fast, and good enough for cosine similarity.
    # we're not doing retrieval here — just measuring semantic distance.
    # if you want better embeddings, swap in a larger model. your call.
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    contexts = list(df["context"].fillna(""))
    utterances = list(df["utterance"])

    print(f"Encoding {len(utterances)} pairs...")
    ctx_embeddings = model.encode(contexts, show_progress_bar=True)
    utt_embeddings = model.encode(utterances, show_progress_bar=True)

    # L = cosine similarity between context and utterance
    # numpy vectorized because for loops are a manner violation
    dot_products = np.sum(ctx_embeddings * utt_embeddings, axis=1)
    ctx_norms = np.linalg.norm(ctx_embeddings, axis=1)
    utt_norms = np.linalg.norm(utt_embeddings, axis=1)
    L = dot_products / (ctx_norms * utt_norms + 1e-8)

    # O = off-topic-ness. higher = more semantically distant from the question.
    O = 1 - L

    df["L_coherence"] = np.round(L, 4)
    df["O_offtopic"] = np.round(O, 4)

    # print summary by maxim and violation_type
    print("\n" + "=" * 60)
    print("COHERENCE SCORES BY MAXIM")
    print("=" * 60)
    for maxim in sorted(df["maxim"].unique()):
        subset = df[df["maxim"] == maxim]
        print(f"\n  {maxim} (n={len(subset)}):")
        print(f"    L (coherence):  mean={subset['L_coherence'].mean():.3f}  std={subset['L_coherence'].std():.3f}")
        print(f"    O (off-topic):  mean={subset['O_offtopic'].mean():.3f}  std={subset['O_offtopic'].std():.3f}")

        # break down by violation_type within each maxim
        if "violation_type" in df.columns:
            for vtype in sorted(subset["violation_type"].unique()):
                vsubset = subset[subset["violation_type"] == vtype]
                print(f"      {vtype:<12} (n={len(vsubset):>2}): L={vsubset['L_coherence'].mean():.3f} ± {vsubset['L_coherence'].std():.3f}")

    # specifically flag Relation examples where annotation might be wrong
    if "violation_type" in df.columns:
        relation = df[df["maxim"] == "Relation"]
        if len(relation) > 0:
            print("\n" + "=" * 60)
            print("RELATION EXAMPLES — SORTED BY COHERENCE (candidates for relabeling)")
            print("=" * 60)
            relation_sorted = relation.sort_values("L_coherence", ascending=False)
            for _, row in relation_sorted.iterrows():
                flag = ""
                # high coherence + flouting = suspicious
                if row["violation_type"] == "flouting" and row["L_coherence"] > 0.4:
                    flag = " ← HIGH coherence for flouting?"
                # low coherence + violating = expected, but very low might be misannotated
                print(f"  L={row['L_coherence']:.3f} O={row['O_offtopic']:.3f} "
                      f"[{row['violation_type']:<12}] "
                      f"{str(row['utterance'])[:50]:<50} / {str(row['context'])[:40]}{flag}")

    # write output
    if output_path:
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"\nScores written to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute semantic coherence scores for annotation quality checking.",
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent.parent / "data" / "annotated" / "corpus.csv"),
        help="Path to annotated CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write scored CSV.",
    )
    args = parser.parse_args()
    compute_coherence(args.data, args.output)
