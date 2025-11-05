import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# Model setup
# -----------------------------
MODEL_NAME = "kirankumarpetlu/Fine_Tunned_LLM"

# Get Hugging Face token from environment (optional if model is private)
HF_TOKEN = os.getenv("HF_TOKEN")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print("Error loading model:", e)
    raise e

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Resume Agent API")

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 200

@app.post("/generate")
def generate_text(request: TextRequest):
    try:
        result = generator(request.prompt, max_length=request.max_length, do_sample=True)
        return {"generated_text": result[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
