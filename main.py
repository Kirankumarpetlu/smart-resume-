from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI(title="Resume LLM API")

# -----------------------------
# Load model (on startup)
# -----------------------------
MODEL_NAME = "kirankumarpetlu/Fine_Tunned_LLM"
TOKEN = "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"

# Load tokenizer and model in 8-bit to save memory
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_auth_token=TOKEN,
    device_map="auto",
    load_in_8bit=True
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# -----------------------------
# Request schema
# -----------------------------
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/generate")
def generate_text(request: PromptRequest):
    outputs = generator(request.prompt, max_new_tokens=request.max_tokens)
    return {"generated_text": outputs[0]["generated_text"]}
