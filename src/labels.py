"""
labels.py

The maxim label schema. The part of the project where I had to commit to
a taxonomy and feel uneasy about it HAHAHAHA

Grice proposed four maxims in 1975 and I've been turning them into a
classification problem, which is either a completely natural thing to do
or a category error depending on who you ask. (Levinson would probably
have notes but am i asking him? naw)

The basic structure: cooperative speech is the baseline n everything else
is a deviation which is either accidental (violating) or deliberate (flouting). 
The interesting cases are the flouting ones, where the deviation IS the
meaning. "the weather is nice today" in response to "why were you late"
a choice. the whole field of pragmatics is basically
just: what do you do with that choice ??

This file also contains the zero-shot hypotheses. Writing natural-language
descriptions of abstract linguistic categories for a BART-MNLI model felt
a little like explaining a joke to a room of people who aren't "in": you either
get it or you don't, and the model mostly doesn't, but it gets closer
than nothing. twas a fun exercise.
"""

from dataclasses import dataclass

# The five labels. Four maxims + Cooperative as the unmarked baseline.
# Ordering matters for LABEL2ID in dataset.py so don't rearrange this
# without also fixing that. (Ask me how I know.)
MAXIMS = ["Cooperative", "Quantity", "Quality", "Relation", "Manner"]

# Secondary label: HOW the maxim is being violated, if it is.
# Flouting is doing it on purpose to mean something.
# Violating is just... failing. Which happens.
# Opting out is when someone says "I can't answer that" — technically
# cooperative in the meta sense but locally non-cooperative. Tricky.
# Clash is when two maxims conflict and you have to break one to satisfy
# the other. Less common but interesting when it shows up.
# NOTE: clash is unreachable in the zero-shot path because multi_label=False
# forces a single winner. it's here for fine-tuned models and human annotation.
# Failed_flout is the gray zone: speaker might be deflecting on purpose but
# there's no surface signal distinguishing it from genuine irrelevance. the
# L+O coherence scoring pass is the empirical check — low coherence between
# Q and A suggests the non-sequitur isn't doing communicative work, which
# means it's either violating or a failed attempt at flouting. either way
# it's not the clean implicature-generating move that "flouting" implies.
#
# six flavors of conversational misbehavior (plus "none" for the well-behaved
# and "unknown" for when the model genuinely can't tell — which is most of the
# time in zero-shot. honesty is a quality maxim thing.)
VIOLATION_TYPES = ["none", "flouting", "violating", "failed_flout", "opting_out", "clash", "unknown"]

# Human-readable descriptions. These are for annotation documentation
# and for anyone reading this who hasn't committed Grice (1975) to memory,
# which is presumably most people.
MAXIM_DESCRIPTIONS = {
    "Cooperative": (
        "The utterance is relevant, truthful, appropriately informative, and clear. "
        "Nobody's doing anything weird. This is the boring label. "
        "You still need it or the classifier has nothing to contrast against."
    ),
    "Quantity": (
        "Too much or too little information given the context. "
        "Classic flouting: say 'some' when you could say 'all' and let the listener "
        "do the math. The scalar implicature cases live here. "
        "Also includes when someone answers a yes/no question with a paragraph — "
        "that's quantity too, just in the other direction."
    ),
    "Quality": (
        "Saying something false, or something you don't have evidence for. "
        "The most morally loaded maxim. Violating quality = lying. "
        "Flouting quality = irony, metaphor, hyperbole. "
        "'I've told you a million times' is a quality flouting. Nobody has said "
        "anything a million times. That's the point."
    ),
    "Relation": (
        "The utterance is not relevant to the current discourse purpose. "
        "The most interesting maxim to flout. An apparent non-sequitur can implicate "
        "almost anything — refusal, discomfort, a topic change that IS the message. "
        "'The weather is nice today' in response to a direct question is doing a lot of work."
    ),
    "Manner": (
        "Obscure, ambiguous, unnecessarily long, or disordered. "
        "The stylistic maxim. Easiest to violate accidentally. "
        "Hardest to detect programmatically because 'unnecessarily long' requires "
        "a model of what 'necessary' means for a given context, which... "
        "well. That's the whole problem, isn't it."
    ),
}

# These are the hypotheses we pass to the NLI model for zero-shot inference.
# Each one is a natural language description of what it would mean for an
# utterance to fall into that category. They're deliberately plain — the model
# doesn't know what a 'maxim' is and I'd rather not find out what it does
# with the word 'Gricean.'
#
# These took several iterations. The first attempt was too theory-laden
# ("this utterance violates the cooperative principle as described by Grice 1975").
# The model did not care. Lesson learned.
ZS_HYPOTHESES = {
    "Cooperative": "This response is relevant, honest, and appropriately informative.",  # the control group
    "Quantity": "This response gives too much or too little information.",  # goldilocks violations
    "Quality": "This response says something false or unsubstantiated.",  # lies and/or literature
    "Relation": "This response changes the subject and does not answer the question.",  # the star of the show tbh
    "Manner": "This response is unclear, ambiguous, or unnecessarily long.",  # the problem child
}


@dataclass
class MaximPrediction:
    """
    The output of a single classification.

    predicted_maxim: the winning label
    violation_type: whether it's flouting, violating, etc. (secondary label)
    confidence: the score for the top prediction
    all_scores: full distribution across all five maxims, for when you want
                to know how close the runner-up was (often interesting;
                Manner and Relation frequently fight for second place)
    """
    utterance: str
    context: str
    predicted_maxim: str
    violation_type: str
    confidence: float
    all_scores: dict

    def __str__(self):
        # the __repr__ i wish academic papers had
        return (
            f"Utterance : {self.utterance!r}\n"
            f"Context   : {self.context!r}\n"
            f"Maxim     : {self.predicted_maxim} ({self.violation_type})\n"
            f"Confidence: {self.confidence:.1%}\n"
            f"All scores: { {k: f'{v:.2f}' for k, v in self.all_scores.items()} }"
        )
