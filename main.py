from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "kirankumarpetlu/Fine_Tunned_LLM"
HF_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = FastAPI(title="Hire Smart API")

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 200

@app.post("/generate")
def generate_text(req: TextRequest):
    response = generator(req.prompt, max_length=req.max_length, do_sample=True)
    return {"generated_text": response[0]["generated_text"]}

@app.get("/")
def root():
    return {"message": "Hire Smart API is live!"}
