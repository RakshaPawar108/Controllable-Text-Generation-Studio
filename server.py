# server.py

from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from controllable_text_gen_phi2 import generate_styled  # your existing generator


class GenerateRequest(BaseModel):
  text: str
  style: str
  strength: int = 50  # 0–100


class GenerateResponse(BaseModel):
  text: str
  style: str
  strength: int
  result: str


class GenerateMultiRequest(BaseModel):
  text: str
  styles: Optional[List[str]] = None
  strength: int = 50  # use same slider value for all


class GenerateMultiResponse(BaseModel):
  text: str
  strength: int
  results: Dict[str, str]



app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # for dev you can also use ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    result = generate_styled(req.text, req.style, req.strength)
    return GenerateResponse(text=req.text, style=req.style, strength=req.strength, result=result)


@app.post("/generate-multi", response_model=GenerateMultiResponse)
async def generate_multi(req: GenerateMultiRequest):
    styles = req.styles or [
        "formal",
        "casual",
        "enthusiastic",
        "sarcastic",
        "poetic",
        "neutral",
    ]
    results: Dict[str, str] = {}
    for style in styles:
        results[style] = generate_styled(req.text, style, req.strength)
    return GenerateMultiResponse(text=req.text, strength=req.strength, results=results)
