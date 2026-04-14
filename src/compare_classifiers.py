"""
compare_classifiers.py

Run the adversarial set through two classifiers and compare:
1. Fine-tuned RoBERTa (local)
2. Claude API (with exchange-type-aware system prompt)

The point: the fine-tuned model has no concept of exchange type norms.
Claude does. The disagreements are where the interesting pragmatics lives.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python compare_classifiers.py
    python compare_classifiers.py --data ../data/annotated/ambiguous_exchanges.csv
    python compare_classifiers.py --output comparison_results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# fail loud and early. not quietly and confusingly.
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY is not set.")
    print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
    print("Get one at: https://console.anthropic.com/settings/keys")
    sys.exit(1)

import anthropic
import pandas as pd
from predict import predict

# exchange type norms — what "cooperative" looks like in each context.
# these are the standards the utterance is being measured against.
# a response that's fine in casual chat might be a manner violation
# in a technical troubleshooting context.
EXCHANGE_NORMS = {
    "performance_eval": (
        "In a performance evaluation exchange, cooperative responses provide "
        "specific, actionable assessment. Vague praise or criticism without "
        "examples is underinformative. The evaluator and evaluatee both expect "
        "concrete details."
    ),
    "technical": (
        "In a technical troubleshooting exchange, cooperative responses identify "
        "root causes, describe specific steps taken, or ask clarifying questions. "
        "Responses that describe symptoms without diagnosis or say 'it should work' "
        "without evidence are underinformative or potentially false."
    ),
    "status_update": (
        "In a status update exchange, cooperative responses give concrete progress "
        "indicators — what's done, what's blocked, what's next. 'Making progress' "
        "without specifics is underinformative. 'On track' without evidence may "
        "be false."
    ),
    "personal": (
        "In a personal check-in exchange, cooperative responses acknowledge the "
        "question and share to the degree the speaker is comfortable. Deflection "
        "may be opting out (legitimate) rather than violating. The norm is less "
        "rigid than professional exchanges."
    ),
    "decision": (
        "In a decision-making exchange, cooperative responses take a position or "
        "articulate trade-offs. 'Either way works' may be genuine flexibility or "
        "underinformative avoidance. The expectation is that participants contribute "
        "to the decision."
    ),
    "knowledge": (
        "In a knowledge exchange, cooperative responses provide accurate, clear "
        "explanations at an appropriate level of detail. Vague gestures at the "
        "answer ('it has to do with how it works') are manner violations. Wrong "
        "answers are quality violations."
    ),
    "conflict": (
        "In a conflict exchange, cooperative responses address the disagreement "
        "directly. 'That's not how I remember it' without providing the alternative "
        "memory is underinformative. Reframing what was said may be quality "
        "violation or legitimate clarification."
    ),
    "instruction": (
        "In an instruction exchange, cooperative responses provide clear, ordered "
        "steps the recipient can follow. 'It's pretty straightforward' without "
        "actual steps is a manner violation. 'Follow the doc' without specifying "
        "which doc is underinformative."
    ),
}

client = anthropic.Anthropic()


def classify_with_claude(utterance: str, context: str, exchange_type: str) -> dict:
    """
    Ask Claude to classify a single utterance with exchange-type context.
    Returns the classification, reasoning, and whether surface proxies were cited.
    """
    norm = EXCHANGE_NORMS.get(exchange_type, "No specific norm defined.")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=(
            "You are evaluating whether the following utterance violates "
            f"a Gricean maxim. The exchange type is {exchange_type}. "
            f"The clarity norm for this exchange type is: {norm} "
            "Classify: Cooperative / Quantity / Quality / Relation / Manner. "
            "Explain your reasoning in 2-3 sentences. "
            "Then state whether your reasoning relied on any surface-level "
            "linguistic markers (caps, scare quotes, hedging language, etc.) "
            "Answer the surface proxy question with just 'yes' or 'no'."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Context: {context}\n"
                f"Utterance: {utterance}\n\n"
                "Classification:"
            ),
        }],
    )

    text = response.content[0].text

    # parse the response. claude is usually structured enough that we can
    # extract the label from the first line. if not, we'll flag it.
    lines = text.strip().split("\n")
    label = "unknown"
    for candidate in ["Cooperative", "Quantity", "Quality", "Relation", "Manner"]:
        if candidate.lower() in lines[0].lower():
            label = candidate
            break

    # surface proxy yes/no is in the raw text — hand-code it from
    # claude_reasoning rather than parsing it here. programmatic
    # extraction was brittle and wrong half the time.

    return {
        "claude_label": label,
        "claude_reasoning": text,
    }


def compare(csv_path: str, output_path: str = None):
    """
    Run both classifiers on the adversarial set and compare.
    """
    df = pd.read_csv(csv_path)
    results = []
    n = len(df)

    print(f"Running {n} pairs through RoBERTa + Claude...\n")

    for i, row in df.iterrows():
        utterance = str(row["utterance"])
        context = str(row["context"])
        exchange_type = str(row.get("exchange_type", "unknown"))

        # 1. fine-tuned RoBERTa
        roberta_pred = predict(utterance, context)

        # 2. Claude API
        claude_pred = classify_with_claude(utterance, context, exchange_type)

        agree = roberta_pred["predicted_maxim"] == claude_pred["claude_label"]

        result = {
            "utterance": utterance,
            "context": context,
            "exchange_type": exchange_type,
            "gold_maxim": row.get("maxim", ""),
            "roberta_label": roberta_pred["predicted_maxim"],
            "roberta_confidence": f"{roberta_pred['confidence']:.3f}",
            "claude_label": claude_pred["claude_label"],
            "agree": agree,
            "claude_reasoning": claude_pred["claude_reasoning"],
        }
        results.append(result)

        status = "✓ agree" if agree else "✗ DISAGREE"
        print(
            f"  [{i+1}/{n}] {utterance[:40]:<40} "
            f"R={roberta_pred['predicted_maxim']:<12} "
            f"C={claude_pred['claude_label']:<12} "
            f"{status}"
        )

        # rate limiting. don't anger the API gods.
        time.sleep(0.5)

    # summary
    agreements = sum(1 for r in results if r["agree"])
    print(f"\nAgreement: {agreements}/{n} ({agreements/n:.1%})")

    # disagreement breakdown
    disagreements = [r for r in results if not r["agree"]]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for r in disagreements:
            print(
                f"  {r['utterance'][:40]:<40} "
                f"R={r['roberta_label']:<12} C={r['claude_label']:<12} "
                f"gold={r['gold_maxim']}"
            )

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RoBERTa and Claude classifiers on adversarial examples.",
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent.parent / "data" / "annotated" / "ambiguous_exchanges.csv"),
        help="Path to adversarial CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write comparison results CSV.",
    )
    args = parser.parse_args()
    compare(args.data, args.output)
