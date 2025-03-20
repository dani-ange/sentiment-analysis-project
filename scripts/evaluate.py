import torch
from transformers import pipeline, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Load dataset (limit to 10 samples for faster evaluation)
dataset = load_dataset("allocine")["test"].select(range(10))

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load model and tokenizer
model_path = "./models"
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=device)

# Load model to get dynamic label mapping
model = AutoModelForSequenceClassification.from_pretrained(model_path)
label_map = model.config.label2id  # Correct direct mapping {LABEL_X: int}

# Get predictions
predictions = [classifier(text["review"], truncation=True, max_length=512)[0]["label"] for text in dataset]
labels = dataset["label"]

# Convert labels (direct mapping)
predictions = [label_map[p] for p in predictions]  

# Compute metrics
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
