from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_intent_model').to('cuda')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Forward map (label to ID, for training)
label_map = {"increase": 0, "decrease": 1, "stop": 2, "set_speed": 3, "change_direction": 4}

# Reverse map (ID to label, for prediction)
id_to_label = {v: k for k, v in label_map.items()}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    intent_id = torch.argmax(outputs.logits, dim=1).item()
    intent = id_to_label.get(intent_id, "unknown")  # Default to "unknown" if ID is out of range
    return intent

if __name__ == "__main__":
    while True:
        text = input("Enter command (or 'exit'): ")
        if text.lower() == 'exit':
            break
        print(f"Predicted intent for '{text}': {predict_intent(text)}")