"""
FastAPI integration example for Arbiter.

This example shows how to:
1. Create an evaluation endpoint
2. Handle async evaluation
3. Return structured results
4. Handle errors gracefully

Run with:
    pip install fastapi uvicorn
    uvicorn examples.fastapi_integration:app --reload

Test with:
    curl -X POST http://localhost:8000/evaluate \
        -H "Content-Type: application/json" \
        -d "{\"output\": \"Paris is the capital of France\", \"reference\": \"The capital of France is Paris\"}"
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from arbiter_ai import evaluate, compare

app = FastAPI(title="Arbiter Evaluation API")


class EvaluateRequest(BaseModel):
    output: str
    reference: Optional[str] = None
    criteria: Optional[str] = None
    evaluators: List[str] = ["semantic"]
    model: str = "gpt-4o-mini"
    threshold: float = 0.7


class EvaluateResponse(BaseModel):
    overall_score: float
    passed: bool
    scores: dict
    cost: float
    processing_time: float


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(request: EvaluateRequest):
    """Evaluate an LLM output."""
    try:
        result = await evaluate(
            output=request.output,
            reference=request.reference,
            criteria=request.criteria,
            evaluators=request.evaluators,
            model=request.model,
            threshold=request.threshold,
        )

        cost = await result.total_llm_cost()

        return EvaluateResponse(
            overall_score=result.overall_score,
            passed=result.passed,
            scores={s.name: s.value for s in result.scores},
            cost=cost,
            processing_time=result.processing_time,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation failed: {str(e)}"
        )


class CompareRequest(BaseModel):
    output_a: str
    output_b: str
    criteria: Optional[str] = None
    model: str = "gpt-4o-mini"


class CompareResponse(BaseModel):
    winner: str
    confidence: float
    reasoning: str


@app.post("/compare", response_model=CompareResponse)
async def compare_endpoint(request: CompareRequest):
    """Compare two LLM outputs."""
    try:
        result = await compare(
            output_a=request.output_a,
            output_b=request.output_b,
            criteria=request.criteria,
            model=request.model,
        )

        return CompareResponse(
            winner=result.winner,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Comparison failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
