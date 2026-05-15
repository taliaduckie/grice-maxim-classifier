import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import predict
from labels import MAXIMS

FEEDBACK_PATH = Path(__file__).parent.parent / "data" / "feedback" / "corrections.csv"

app = FastAPI(title="Grice Maxim Classifier", version="1.0.0")

# CORS so a frontend can hit this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict before prod
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    utterance: str
    context: str = ""


class ClassifyResponse(BaseModel):
    utterance: str
    context: str
    predicted_maxim: str
    violation_type: str
    confidence: float
    all_scores: dict[str, float]
    low_confidence: bool


class BatchRequest(BaseModel):
    pairs: list[ClassifyRequest]


class CorrectionRequest(BaseModel):
    utterance: str
    context: str = ""
    corrected_maxim: str
    notes: Optional[str] = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.utterance.strip():
        raise HTTPException(status_code=400, detail="utterance cannot be empty")

    result = predict(req.utterance, req.context)
    return ClassifyResponse(
        utterance=req.utterance,
        context=req.context,
        predicted_maxim=result["predicted_maxim"],
        violation_type=result["violation_type"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        low_confidence=result["confidence"] < 0.7,
    )


@app.post("/batch")
def batch_classify(req: BatchRequest):
    if len(req.pairs) > 500 or len(req.pairs) == 0:
        raise HTTPException(400, "pairs must be between 1 and 500")

    results = []
    for pair in req.pairs:
        result = predict(pair.utterance, pair.context)
        results.append({
            "utterance": pair.utterance,
            "context": pair.context,
            "predicted_maxim": result["predicted_maxim"],
            "violation_type": result["violation_type"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "low_confidence": result["confidence"] < 0.7,
        })
    return {"results": results, "count": len(results)}


@app.post("/correct")
def submit_correction(req: CorrectionRequest):
    if not req.utterance.strip():
        raise HTTPException(400, "utterance cannot be empty")
    if req.corrected_maxim not in MAXIMS:
        raise HTTPException(400, f"corrected_maxim must be one of {MAXIMS}")

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_PATH.exists()

    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "utterance", "context", "corrected_maxim", "notes", "timestamp",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "utterance": req.utterance,
            "context": req.context,
            "corrected_maxim": req.corrected_maxim,
            "notes": req.notes or "",
            "timestamp": datetime.now().isoformat(),
        })

    return {"status": "saved", "message": f"saved correction: {req.corrected_maxim}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
