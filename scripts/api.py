"""
BabyPix1 FastAPI server for text generation via HTTP.

Usage:
    uvicorn scripts.api:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from babypix1.generator import TextGenerator

app = FastAPI(title="BabyPix1 Generation API")

generator = TextGenerator.from_checkpoint("checkpoints/best_model.pt", "tokenizer_quick.json")


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str


@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    result = generator.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
    return GenerationResponse(prompt=request.prompt, generated_text=result)