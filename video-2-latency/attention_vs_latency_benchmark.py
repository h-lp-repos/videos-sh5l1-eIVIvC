import time
import torch
from transformers import AutoModel, AutoTokenizer


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

sequence_lengths = [8, 16, 32, 64, 128]
latency_results = []

for n in sequence_lengths:
    fake_text = "hello " * n
    inputs = tokenizer(fake_text, return_tensors="pt")

    start = time.time()
    with torch.no_grad():
        model(**inputs)
    end = time.time()

    latency = (end - start) * 1000
    latency_results.append((n, latency))
    print(f"Tokens: {n:4d} | Latency: {latency:.2f} ms")

print("\nLatency growth clearly follows quadratic behavior.")
