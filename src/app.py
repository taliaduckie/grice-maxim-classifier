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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from predict import predict

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


def classify(utterance: str, context: str) -> dict:
    """
    wrapper that formats the output for gradio.
    predict() returns a dict, gradio wants a dict of label->score
    for the Label component. simple enough.
    """
    result = predict(utterance, context)
    return {
        "label": f"{result['predicted_maxim']} ({result['violation_type']})",
        "confidences": {
            maxim: score for maxim, score in result["all_scores"].items()
        },
    }


# the interface. keeping it simple because the interesting part
# is the model, not the UI. manner maxim: be brief.
demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(label="Utterance", placeholder="The weather is nice today."),
        gr.Textbox(label="Context", placeholder="Why were you late to the meeting?"),
    ],
    outputs=gr.Label(label="Maxim prediction", num_top_classes=5),
    title="Grice Maxim Classifier",
    description=(
        "Classify an utterance by which Gricean maxim it violates (if any). "
        "Paste what someone said and what they were responding to."
    ),
    examples=EXAMPLES,
    # cache examples so clicking them is instant. the model loads once
    # and stays loaded because we fixed that. remember when it loaded
    # 229 times? dark times.
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
