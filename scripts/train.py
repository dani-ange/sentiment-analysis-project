from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Load dataset (French dataset example: Allociné)
dataset = load_dataset("allocine")

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
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_steps=1000,
    load_best_model_at_end=True,
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
