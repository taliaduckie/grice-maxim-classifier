"""
bootstrap.py

Use the zero-shot model to pre-label a bunch of utterance-context pairs
so you don't have to annotate from scratch like some kind of animal.

the idea: write down a bunch of examples that probably violate various maxims,
run the classifier on them, dump a CSV with the model's guesses, then go
through and fix the ones it got wrong. which it will. but fixing is faster
than labeling from nothing, and that's the whole game.

the seed examples here are biased toward flouting because flouting is
interesting and violating is just... sad. i'll add more violating examples
later when i'm in a worse mood.

Usage:
    python bootstrap.py
    python bootstrap.py --output ../data/annotated/bootstrap_labeled.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# same sys.path incantation. at this point it's a tradition.
sys.path.insert(0, str(Path(__file__).parent))

from zero_shot import classify

# utterance-context pairs organized by what i THINK they are.
# the model may disagree. that's the point. disagreement is data.
SEED_PAIRS = [
    # --- Relation flouting ---
    # the classic "i'm not going to answer your question and we both know it" move
    ("The weather is nice today.", "Why were you late to the meeting?"),
    ("Have you tried the new coffee place?", "Did you finish the report?"),
    ("My cat did the funniest thing yesterday.", "Can we talk about the budget?"),
    ("I hear it might rain tomorrow.", "Are you going to apologize?"),
    ("That's a nice shirt.", "Did you read my email?"),
    ("How about those playoffs?", "We need to discuss your performance review."),
    ("I'm thinking about getting a new plant.", "When are you going to pay me back?"),

    # --- Quantity flouting ---
    # saying less than you know, or more than anyone asked for
    ("Some students passed.", "Did everyone pass the exam?"),
    ("I've been to a country or two.", "Have you traveled much?"),
    ("She's not the worst singer.", "What did you think of her performance?"),
    ("Well, where do I even begin. So first I woke up at 6:47, not 6:45 like usual, and then I brushed my teeth for exactly two minutes...", "How was your morning?"),
    ("He has a pulse.", "How's the new intern doing?"),
    ("There were some issues.", "How did the deployment go?"),

    # --- Quality flouting ---
    # irony, hyperbole, metaphor — saying something false ON PURPOSE
    ("Oh sure, I LOVE waiting in line for three hours.", "How do you feel about the DMV?"),
    ("I've told you a million times.", "Can you remind me of the password?"),
    ("He's a real Einstein.", "What do you think of the new hire?"),
    ("My heart literally exploded.", "How was the concert?"),
    ("Yeah and I'm the Queen of England.", "I finished the entire project last night."),
    ("That went well.", "The presentation crashed halfway through and the client left."),

    # --- Quality violating ---
    # actually getting things wrong, not on purpose. just... wrong.
    ("The capital of Australia is Sydney.", "What's the capital of Australia?"),
    ("I'm pretty sure the meeting is at 3.", "When is the meeting? It's at 2."),
    ("That word means happy.", "What does 'lugubrious' mean?"),

    # --- Manner flouting/violating ---
    # the sin of being unclear, wordy, or weirdly structured
    ("I may or may not have potentially been in a situation where something could have occurred.", "What happened?"),
    ("The thing with the stuff at the place.", "What are you talking about?"),
    ("First I did step 3, then step 1, then I went back to step 2.", "How did you assemble the shelf?"),
    ("She performed a series of bilateral contractions of the orbicularis oculi muscles.", "What did she do?"),
    ("It's not not possible that it's not untrue.", "Is this correct?"),

    # --- Cooperative ---
    # boring but necessary. the control group needs love too.
    ("The meeting is at 3pm in room 204.", "When and where is the meeting?"),
    ("I finished the report and sent it to Sarah.", "What did you do today?"),
    ("Yes, I'll be there.", "Can you make it to dinner tonight?"),
    ("It costs forty-five dollars.", "How much is it?"),
    ("I left them on the kitchen counter.", "Have you seen my keys?"),
    ("No, I haven't heard from her since Tuesday.", "Have you talked to Maria?"),
    ("The train arrives at 5:15.", "When does the train get here?"),
]


def bootstrap(output_path: str):
    """
    run zero-shot on all seed pairs and dump to CSV.
    the 'gold' column is empty — that's for you, human annotator.
    the 'predicted' column is what the model thinks. argue with it.
    """
    results = []

    print(f"Running zero-shot on {len(SEED_PAIRS)} pairs...")
    print("(this loads the model once and then it's fast. ish.)\n")

    for i, (utterance, context) in enumerate(SEED_PAIRS, 1):
        pred = classify(utterance, context)
        results.append({
            "utterance": utterance,
            "context": context,
            "predicted_maxim": pred.predicted_maxim,
            "predicted_violation_type": pred.violation_type,
            "confidence": f"{pred.confidence:.3f}",
            # leave these blank for the human to fill in.
            # the whole point is you look at predicted_maxim and go
            # "yeah" or "no actually that's Relation you fool"
            "gold_maxim": "",
            "gold_violation_type": "",
        })
        # progress dots because waiting in silence is a manner violation
        print(f"  [{i}/{len(SEED_PAIRS)}] {utterance[:50]:<50} → {pred.predicted_maxim} ({pred.confidence:.0%})")

    # write it out
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} pre-labeled examples to {output_path}")
    print("Now go fill in gold_maxim and gold_violation_type.")
    print("When you're done, drop the predicted columns and rename gold → maxim/violation_type")
    print("to match corpus.csv format. or don't and i'll write a script for that too.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap annotation by pre-labeling with zero-shot model.",
        epilog="The model will be wrong sometimes. That's the point. You fix it."
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "annotated" / "bootstrap_labeled.csv"),
        help="Where to write the pre-labeled CSV.",
    )
    args = parser.parse_args()
    bootstrap(args.output)
