import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()

# Input sentence
sentence = "Despite his injury, the striker scored the winning goal."
inputs = tokenizer(sentence, return_tensors="pt")

# Forward pass with attention output
with torch.no_grad():
    outputs = model(**inputs)

# Extract attention from last layer, first head
attention = outputs.attentions[-1][0][0].detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Plot attention heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
plt.title("Self-Attention Visualization")
plt.xlabel("Keys")
plt.ylabel("Queries")
plt.tight_layout()

# Save image
output_path = "outputs/self_attention_heatmap.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Attention heatmap saved at: {output_path}")
