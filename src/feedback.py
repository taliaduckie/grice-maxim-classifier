"""Recording user corrections.

app.py and api.py both appended to data/feedback/corrections.csv with different
column sets — the Gradio app wrote four columns, the API wrote five including a
timestamp. Whichever ran first defined the header, and the other then appended
rows of the wrong width to it.
"""

import csv
from datetime import datetime

from labels import MAXIMS
from paths import CORRECTIONS_PATH

COLUMNS = ["utterance", "context", "corrected_maxim", "notes", "timestamp"]


def record_correction(utterance, context="", corrected_maxim="", notes="",
                      path=CORRECTIONS_PATH):
    """Append one correction. Raises ValueError on invalid input.

    Callers translate the exception into whatever their transport needs — an
    HTTP 400 for the API, a message string for the Gradio app.
    """
    if not (utterance or "").strip():
        raise ValueError("utterance cannot be empty")
    if corrected_maxim not in MAXIMS:
        raise ValueError(f"corrected_maxim must be one of {MAXIMS}")

    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "utterance": utterance,
            "context": context or "",
            "corrected_maxim": corrected_maxim,
            "notes": notes or "",
            "timestamp": datetime.now().isoformat(),
        })
    return path
