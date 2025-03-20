from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
dataset = load_dataset("allocine")["test"]

# Load model
classifier = pipeline("text-classification", model="./models")

# Get predictions
predictions = [classifier(text["review"])[0]["label"] for text in dataset]
labels = dataset["label"]

# Convert labels
label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
predictions = [label_map[p] for p in predictions]

# Compute metrics
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
