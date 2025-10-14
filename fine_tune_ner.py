import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from seqeval.metrics import classification_report

# Load dataset
dataset = load_dataset('csv', data_files='ner_commands.csv')
dataset = dataset['train'].train_test_split(test_size=0.2)

# Define labels
id2label = {0: "O", 1: "B-VALUE", 2: "I-VALUE", 3: "B-UNIT", 4: "I-UNIT", 5: "B-DIRECTION", 6: "I-DIRECTION"}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(examples):
    # Tokenize sentences
    tokenized_inputs = tokenizer(
        examples['sentence'],
        truncation=True,
        padding='max_length',
        max_length=32,
        is_split_into_words=False
    )
    labels = []
    
    for i, (sentence, label_str) in enumerate(zip(examples['sentence'], examples['labels'])):
        # Split sentence and labels manually
        sentence_tokens = sentence.split()
        label_list = label_str.strip().split()
        
        # Check alignment
        if len(sentence_tokens) != len(label_list):
            print(f"Warning: Mismatch in sentence {i+1}: '{sentence}'")
            print(f"Sentence tokens ({len(sentence_tokens)}): {sentence_tokens}")
            print(f"Labels ({len(label_list)}): {label_list}")
            labels.append([-100] * 32)  # Skip with dummy labels
            continue
        
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        label_idx = 0
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens ([CLS], [SEP])
            elif word_idx != previous_word_idx:
                # New word: assign label
                if label_idx < len(label_list):
                    try:
                        label_ids.append(label2id[label_list[label_idx]])
                        label_idx += 1
                    except KeyError as e:
                        print(f"Error: Invalid label '{label_list[label_idx]}' in sentence '{sentence}'")
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
            else:
                # Subword token
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        # Ensure label_ids matches max_length
        if len(label_ids) < 32:
            label_ids.extend([-100] * (32 - len(label_ids)))
        elif len(label_ids) > 32:
            label_ids = label_ids[:32]
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Model
model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels, id2label=id2label, label2id=label2id)
model.to('cuda')

# Training args
training_args = TrainingArguments(
    output_dir='./ner_results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./ner_logs',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    save_total_limit=2,
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    true_labels = [[id2label[l] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    flat_preds = [item for sublist in true_preds for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    return {
        'accuracy': accuracy_score(flat_labels, flat_preds),
        'f1': f1_score(flat_labels, flat_preds, average='weighted'),
        'seqeval_report': classification_report(true_labels, true_preds, output_dict=True)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('./fine_tuned_ner_model')
tokenizer.save_pretrained('./fine_tuned_ner_model')