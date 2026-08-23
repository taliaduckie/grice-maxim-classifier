"""Every filesystem location the project uses.

Ten modules were each deriving ROOT from __file__ and rebuilding the same
paths, which is how corpus.csv stayed the default in three scripts after
corpus_train.csv became the real training set.

Layout:
    data/annotated/   hand-maintained sources — edited by people
    data/derived/     regenerated from sources by a script; safe to delete
    data/test/        the frozen test set (see test/manifest.json)
    data/raw/         scraper output
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data"
ANNOTATED_DIR = DATA_DIR / "annotated"
DERIVED_DIR = DATA_DIR / "derived"
RAW_DIR = DATA_DIR / "raw"
TEST_DIR = DATA_DIR / "test"
FEEDBACK_DIR = DATA_DIR / "feedback"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

# --- sources: edited by hand, never regenerated ---
CORPUS_PATH = ANNOTATED_DIR / "corpus.csv"
SYNTHETIC_PATH = ANNOTATED_DIR / "corpus_367.csv"
AMBIGUOUS_PATH = ANNOTATED_DIR / "ambiguous_exchanges.csv"

# --- derived: written by a script, reproducible ---
TRAIN_PATH = DERIVED_DIR / "corpus_train.csv"
PROVENANCE_PATH = DERIVED_DIR / "corpus_provenance.csv"
QA_PAIRS_PATH = DERIVED_DIR / "natural_qa_pairs.csv"

# --- frozen test set ---
TEST_NATURAL = TEST_DIR / "test_natural.csv"
TEST_SYNTHETIC = TEST_DIR / "test_synthetic.csv"
TEST_PENDING = TEST_DIR / "test_natural_pending.csv"
MANIFEST_PATH = TEST_DIR / "manifest.json"

# --- other ---
MODEL_DIR = MODELS_DIR / "roberta-grice"
CORRECTIONS_PATH = FEEDBACK_DIR / "corrections.csv"


def rel(path) -> str:
    """Path relative to the repo root, for printing."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
