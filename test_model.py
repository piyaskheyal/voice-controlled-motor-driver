from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_intent_model').to('cuda')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
label_map = {0: "increase", 1: "decrease", 2: "stop", 3: "change_direction"}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32).to('cuda')
    outputs = model(**inputs)
    intent_id = torch.argmax(outputs.logits, dim=1).item()
    return label_map[intent_id]

# Test
print(predict_intent("Change direction"))  # Expected: change_direction