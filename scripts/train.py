from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load dataset (French dataset example: Allociné)
dataset = load_dataset("allocine")
dataset["train"] = dataset["train"].select(range(10))  # Train on 500 samples
dataset["test"] = dataset["test"].select(range(5))  # Test on 200 samples
# Load tokenizer
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize data
def tokenize(batch):
    return tokenizer(batch["review"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",  # Ensure this matches the save_strategy
    save_strategy="epoch",  # Change this to "epoch" to match evaluation_strategy
    load_best_model_at_end=True,  # Ensures best model is loaded
    save_total_limit=2,  # Keep only the last 2 models to save space
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)


# Train model
trainer.train()

# Save model
model.save_pretrained("./models")
tokenizer.save_pretrained("./models")
