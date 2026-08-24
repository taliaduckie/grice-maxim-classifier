import csv
import sys
from pathlib import Path


import gradio as gr
import pandas as pd
from predict import predict
from feedback import record_correction

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
### the maxims (very briefly)

Grice (1975) said cooperative speakers follow four maxims:

- **Quantity** — say enough but not too much
- **Quality** — don't say what you believe to be false
- **Relation** — be relevant
- **Manner** — be clear

The fifth label, **Cooperative**, just means none of the above were violated.

### flouting vs violating

Flouting = breaking a maxim on purpose so the listener picks up on it (sarcasm, irony, indirect refusal).
Violating = breaking it by accident or to deceive.

The model can't reliably tell flouting from violating yet — it predicts the maxim only.

### why the model might disagree with you

Pragmatics is subjective. "Fine." could be Quantity (too little) or Manner (vague) depending on what
you think the underlying problem is. Even linguists disagree ~20-30% of the time. If confidence is
below 70%, take it as a starting point not a verdict.
"""


def classify(utterance: str, context: str):
    if not utterance.strip():
        empty_df = pd.DataFrame({"maxim": [], "score": []})
        return "### Enter an utterance", "", empty_df

    result = predict(utterance, context)
    confidence = result["confidence"]
    top = result["predicted_maxim"]

    # big prediction header
    header = f"### Prediction: **{top}** ({result['violation_type']}) — {confidence:.0%}"

    # confidence note
    if confidence < 0.5:
        warning = (
            f"Low confidence. The model is unsure — consider the runner-up labels in the chart."
        )
    elif confidence < 0.7:
        warning = (
            f"Moderate confidence. The model leans toward {top} but isn't highly certain."
        )
    else:
        warning = "High confidence."

    # bar chart data, sorted high to low
    scores = sorted(result["all_scores"].items(), key=lambda x: -x[1])
    df = pd.DataFrame({
        "maxim": [m for m, _ in scores],
        "score": [s for _, s in scores],
    })

    return header, warning, df


def submit_correction(utterance: str, context: str, correct_maxim: str, notes: str) -> str:
    try:
        record_correction(utterance, context, correct_maxim, notes)
    except ValueError as e:
        return f"Please check your input: {e}"
    return "Correction saved. Thank you!"


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
                    prediction_header = gr.Markdown("### Prediction will appear here")
                    warning_output = gr.Textbox(
                        label="Confidence",
                        interactive=False,
                        lines=2,
                    )
                    score_plot = gr.BarPlot(
                        value=pd.DataFrame({"maxim": [], "score": []}),
                        x="score",
                        y="maxim",
                        color="maxim",
                        color_map={
                            "Quality": "#4C72B0",     # blue
                            "Quantity": "#DD8452",    # orange
                            "Relation": "#55A868",    # green
                            "Manner": "#C44E52",      # red
                            "Cooperative": "#8172B3", # purple
                        },
                        title="Score distribution",
                        x_lim=[0, 1],
                        height=300,
                    )

            classify_btn.click(
                fn=classify,
                inputs=[utterance_input, context_input],
                outputs=[prediction_header, warning_output, score_plot],
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
