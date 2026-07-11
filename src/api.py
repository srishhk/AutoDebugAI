from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import score_pr

app = FastAPI(
    title="AutoDebug AI",
    description="Predicts bug risk for GitHub pull requests",
    version="0.1.0",
)


class PRFeatures(BaseModel):
    additions: int
    deletions: int
    changed_files: int
    commits: int
    comments: int = 0
    review_comments: int = 0
    num_files: int
    author_past_prs: int = 0
    author_past_bug_rate: float = 0.0
    is_first_pr: int = 1
    title: str


class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    reasons: list[str]


def risk_level(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


@app.get("/")
def root():
    return {"status": "ok", "service": "AutoDebug AI", "docs": "/docs"}


@app.post("/predict", response_model=RiskResponse)
def predict(pr: PRFeatures):
    try:
        result = score_pr(pr.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "risk_score": result["risk_score"],
        "risk_level": risk_level(result["risk_score"]),
        "reasons": result["reasons"],
    }