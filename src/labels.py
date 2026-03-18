"""
labels.py
Maxim label schema and definitions.
"""

from dataclasses import dataclass

MAXIMS = ["Cooperative", "Quantity", "Quality", "Relation", "Manner"]

VIOLATION_TYPES = ["none", "flouting", "violating", "opting_out", "clash"]

MAXIM_DESCRIPTIONS = {
    "Cooperative": "The utterance is relevant, truthful, appropriately informative, and clear.",
    "Quantity": (
        "The utterance provides more or less information than required. "
        "Flouting quantity can generate scalar implicatures."
    ),
    "Quality": (
        "The utterance asserts something the speaker believes to be false or "
        "lacks adequate evidence for."
    ),
    "Relation": (
        "The utterance is not relevant to the current purpose of the exchange. "
        "Flouting relation often signals implicature via apparent non-sequitur."
    ),
    "Manner": (
        "The utterance is obscure, ambiguous, unnecessarily prolix, or disorderly. "
        "Flouting manner can signal irony or deliberate vagueness."
    ),
}

ZS_HYPOTHESES = {
    "Cooperative": "This response is relevant, honest, and appropriately informative.",
    "Quantity": "This response gives too much or too little information.",
    "Quality": "This response says something false or unsubstantiated.",
    "Relation": "This response is irrelevant to what was asked.",
    "Manner": "This response is unclear, ambiguous, or unnecessarily long.",
}


@dataclass
class MaximPrediction:
    utterance: str
    context: str
    predicted_maxim: str
    violation_type: str
    confidence: float
    all_scores: dict

    def __str__(self):
        return (
            f"Utterance : {self.utterance!r}\n"
            f"Context   : {self.context!r}\n"
            f"Maxim     : {self.predicted_maxim} ({self.violation_type})\n"
            f"Confidence: {self.confidence:.1%}\n"
            f"All scores: { {k: f'{v:.2f}' for k, v in self.all_scores.items()} }"
        )
