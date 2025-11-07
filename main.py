from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# 🔐 Authenticate
login("your_huggingface_token_here")

# Define base + adapter
base_model = "google/gemma-2b-it"
adapter_model = "kirankumarpetlu/Fine_Tunned_LLM"

# Load base Gemma model
tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(base_model, use_auth_token=True)

# Load the adapter on top
model = PeftModel.from_pretrained(model, adapter_model, use_auth_token=True)

# (Optional) merge LoRA weights into the base model
model = model.merge_and_unload()


# After merging
model.save_pretrained("Fine_Tuned_Gemma2_Merged")
tokenizer.save_pretrained("Fine_Tuned_Gemma2_Merged")