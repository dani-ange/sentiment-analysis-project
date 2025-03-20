from transformers import pipeline, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
dataset = load_dataset("allocine")["test"]
dataset["test"] = dataset["test"].select(range(5))  # Test on 200 samples

# Load model and tokenizer
model_path = "./models"
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Get actual model labels
model = AutoModelForSequenceClassification.from_pretrained(model_path)
label_map = {v: k for k, v in model.config.label2id.items()}  # Adjust dynamically

# Get predictions
predictions = [classifier(text["review"], truncation=True, max_length=512)[0]["label"] for text in dataset]
labels = dataset["label"]

# Convert labels
predictions = [label_map[p] for p in predictions]

# Compute metrics
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
