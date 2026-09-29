"""llm_annotate: the model sees what a human sees, and writes what a human writes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llm_annotate as L
from annotate import MAXIM_CHOICES, VTYPE_CHOICES, OUT_COLUMNS, append_annotation


def test_system_prompt_is_the_guidelines():
    s = L.system_prompt()
    assert s.startswith(L.FRAMING)
    assert "## 2. The decision procedure" in s
    assert "## 5. Cases with a fixed convention" in s


def test_schema_matches_annotator_vocabulary():
    assert L.SCHEMA["properties"]["maxim"]["enum"] == MAXIM_CHOICES
    vt = L.SCHEMA["properties"]["violation_type"]["enum"]
    assert set(VTYPE_CHOICES) <= set(vt) and "none" in vt
    assert L.SCHEMA["additionalProperties"] is False


def test_item_prompt_shows_only_the_pair():
    row = {"row_id": "abc", "context": "Q?", "utterance": "A.",
           "subreddit": "askscience", "maxim": "Quality"}
    p = L.item_prompt(row)
    assert "Q?" in p and "A." in p
    assert "askscience" not in p and "Quality" not in p and "abc" not in p


def test_judgment_round_trips_through_annotate_writer(tmp_path):
    out = tmp_path / "claude.csv"
    row = {"row_id": "r1", "context": "c", "utterance": "u"}
    append_annotation(out, row, "Cooperative", "flouting", 4, "")
    text = out.read_text()
    assert text.splitlines()[0] == ",".join(f'"{c}"' for c in OUT_COLUMNS)
    assert '"Cooperative","none"' in text   # §3: none forced for Cooperative
