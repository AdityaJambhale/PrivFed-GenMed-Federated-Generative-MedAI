from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load your final federated model
model = GPT2LMHeadModel.from_pretrained("PrivFedGenMed_Final")
tokenizer = GPT2Tokenizer.from_pretrained("PrivFedGenMed_Final")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Give it a prompt
prompt = "Patient admitted with"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate
outputs = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
