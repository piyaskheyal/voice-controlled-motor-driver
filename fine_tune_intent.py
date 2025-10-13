import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Load dataset (offline)
dataset = load_dataset('csv', data_files='commands.csv')
dataset = dataset['train'].train_test_split(test_size=0.2)  # 80/20 split

# Tokenizer (downloaded offline)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=32)  # Short commands

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Map intents to labels
label_map = {"increase": 0, "decrease": 1, "stop": 2, "change_direction": 3}
tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': label_map[examples['intent']]})

# Model (small, fine-tune on GPU)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='epoch',  # Updated from evaluation_strategy
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_steps=10,  # Log more frequently for small dataset
    save_total_limit=2,  # Save only best 2 checkpoints
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='weighted')}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# Fine-tune (runs on GPU, ~5-10 min)
trainer.train()

# Save for offline use
trainer.save_model('./fine_tuned_intent_model')
tokenizer.save_pretrained('./fine_tuned_intent_model')