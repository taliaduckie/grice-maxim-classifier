"""
app.py

Gradio demo for the Grice maxim classifier.

paste an utterance + context, get a maxim prediction. that's it.
that's the app. no one said it had to be complicated.

Usage:
    python app.py

Then open http://localhost:7860 in your browser. or don't. i'm a
docstring, not a cop.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from predict import predict

FEEDBACK_PATH = Path(__file__).parent.parent / "data" / "feedback" / "corrections.csv"

# examples that show off each maxim. picked these because they're
# fun and also because "The weather is nice today" is basically
# the project's mascot at this point.
EXAMPLES = [
    ["The weather is nice today.", "Why were you late to the meeting?"],
    ["Oh sure, I LOVE waiting in line for three hours.", "How do you feel about the DMV?"],
    ["The meeting is at 3pm in room 204.", "When and where is the meeting?"],
    ["I may or may not have potentially been in a situation where something could have occurred.", "What happened?"],
    ["Some students passed.", "Did everyone pass the exam?"],
    ["React is a programming language.", "What framework are you using?"],
    ["Great, the client loved it. You could really tell from how fast they left.", "How did the presentation land?"],
]

INFO_TEXT = """
## What are Gricean maxims?

In 1975, philosopher Paul Grice proposed that conversation follows an unspoken **Cooperative Principle**:
we generally try to be helpful when we talk to each other. He broke this down into four maxims:

| Maxim | What it means | Example violation |
|---|---|---|
| **Quantity** | Say enough, but not too much | "Fine." (when asked for a detailed explanation) |
| **Quality** | Don't say things that are false or unsubstantiated | "React is a programming language." |
| **Relation** | Be relevant to the topic at hand | "The weather is nice today." (when asked why you were late) |
| **Manner** | Be clear, unambiguous, and orderly | "The thing with the stuff at the place." |

A fifth label, **Cooperative**, means no violation — the speaker is being helpful, truthful, relevant, and clear.

## Flouting vs. Violating

- **Flouting**: breaking a maxim *on purpose* to communicate something indirectly. Sarcasm, irony,
  deliberate understatement — the listener is supposed to notice the violation and infer the real meaning.
- **Violating**: breaking a maxim *accidentally* or *deceptively*. The speaker failed to communicate well,
  or is actively trying to mislead.

## Why the model might disagree with you

Pragmatics is inherently subjective. The same utterance can be Quantity (too little info) or Manner
(unclear expression) depending on whether you think the problem is *what* was said or *how* it was said.
Even trained linguists disagree ~20-30% of the time on maxim labels. If the model's confidence is below
70%, treat the prediction as a starting point for discussion, not a definitive answer.
"""


def classify(utterance: str, context: str) -> tuple:
    """
    wrapper that formats the output for gradio.
    returns the label dict plus a confidence warning if needed.
    """
    if not utterance.strip():
        return {"label": "Enter an utterance", "confidences": {}}, ""

    result = predict(utterance, context)
    confidence = result["confidence"]

    # confidence warning
    if confidence < 0.5:
        warning = (
            f"Low confidence ({confidence:.0%}). The model is unsure about this one — "
            f"the prediction may not be reliable. Consider the runner-up labels."
        )
    elif confidence < 0.7:
        warning = (
            f"Moderate confidence ({confidence:.0%}). The model leans toward "
            f"{result['predicted_maxim']} but isn't highly certain."
        )
    else:
        warning = ""

    label_output = {
        "label": f"{result['predicted_maxim']} ({result['violation_type']})",
        "confidences": {
            maxim: score for maxim, score in result["all_scores"].items()
        },
    }

    return label_output, warning


def submit_correction(utterance: str, context: str, correct_maxim: str, notes: str) -> str:
    """
    save a user correction to a CSV file for later review and
    potential incorporation into the training corpus.
    free annotation data from people who actually care about
    getting it right.
    """
    if not utterance.strip() or not correct_maxim.strip():
        return "Please enter an utterance and select a maxim."

    # ensure feedback directory exists
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance", "context", "corrected_maxim", "notes"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "utterance": utterance,
            "context": context,
            "corrected_maxim": correct_maxim,
            "notes": notes,
        })

    return f"Correction saved. Thank you!"


# build the interface with tabs
with gr.Blocks(title="Grice Maxim Classifier") as demo:
    gr.Markdown("# Grice Maxim Classifier")
    gr.Markdown(
        "Classify an utterance by which Gricean maxim it violates (if any). "
        "Paste what someone said and what they were responding to."
    )

    with gr.Tabs():
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column():
                    utterance_input = gr.Textbox(
                        label="Utterance",
                        placeholder="The weather is nice today.",
                        lines=2,
                    )
                    context_input = gr.Textbox(
                        label="Context (what they were responding to)",
                        placeholder="Why were you late to the meeting?",
                        lines=2,
                    )
                    classify_btn = gr.Button("Classify", variant="primary")

                with gr.Column():
                    label_output = gr.Label(label="Prediction", num_top_classes=5)
                    warning_output = gr.Textbox(
                        label="Confidence note",
                        interactive=False,
                        lines=2,
                    )

            classify_btn.click(
                fn=classify,
                inputs=[utterance_input, context_input],
                outputs=[label_output, warning_output],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[utterance_input, context_input],
            )

        with gr.Tab("Correct a prediction"):
            gr.Markdown(
                "Think the model got it wrong? Submit a correction. "
                "These get saved for review and may improve future versions."
            )
            corr_utterance = gr.Textbox(label="Utterance", lines=2)
            corr_context = gr.Textbox(label="Context", lines=2)
            corr_maxim = gr.Dropdown(
                choices=["Cooperative", "Quantity", "Quality", "Relation", "Manner"],
                label="Correct maxim",
            )
            corr_notes = gr.Textbox(
                label="Notes (optional)",
                placeholder="Why you think this is the right label...",
                lines=2,
            )
            corr_btn = gr.Button("Submit correction")
            corr_result = gr.Textbox(label="Status", interactive=False)

            corr_btn.click(
                fn=submit_correction,
                inputs=[corr_utterance, corr_context, corr_maxim, corr_notes],
                outputs=corr_result,
            )

        with gr.Tab("About"):
            gr.Markdown(INFO_TEXT)

if __name__ == "__main__":
    demo.launch()
