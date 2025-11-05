# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "kirankumarpetlu/Fine_Tunned_LLM"  # Public model, no token needed

# -----------------------------
# Load model & tokenizer
# -----------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Model loaded successfully!")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Resume AI API")

# -----------------------------
# Request model
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/generate")
def generate_text(request: GenerateRequest):
    """
    Generate text from a prompt using the fine-tuned model.
    """
    output = generator(request.prompt, max_length=request.max_length, do_sample=True)
    return {"generated_text": output[0]['generated_text']}

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
