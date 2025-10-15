import spacy
import random
from spacy.training import Example
from spacy.util import minibatch
import pandas as pd
from pathlib import Path

# Check spaCy dependencies
try:
    import spacy_lookups_data
except ImportError:
    print("Error: spacy-lookups-data not installed. Run: pip install spacy-lookups-data")
    exit(1)

# Load data
def load_ner_data(csv_path):
    df = pd.read_csv(csv_path)
    training_data = []
    for _, row in df.iterrows():
        sentence = row['sentence']
        labels = row['labels'].split()
        words = sentence.split()
        if len(words) != len(labels):
            print(f"Warning: Mismatch in sentence: '{sentence}'")
            continue
        entities = []
        start = 0
        for word, label in zip(words, labels):
            end = start + len(word)
            if label.startswith('B-'):
                entity_type = label[2:]  # VALUE, UNIT, DIRECTION
                entities.append((start, end, entity_type))
            elif label.startswith('I-') and entities and entities[-1][2] == label[2:]:
                # Extend the previous entity
                entities[-1] = (entities[-1][0], end, entities[-1][2])
            start = end + 1  # Account for space
        training_data.append((sentence, {"entities": entities}))
    return training_data

# Convert to spaCy format
def convert_to_spacy_format(data):
    return [(text, {"entities": annot["entities"]}) for text, annot in data]

# Load and split data
data = load_ner_data('ner_commands.csv')
random.seed(42)
random.shuffle(data)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")  # or en_core_web_lg for better accuracy
except Exception as e:
    print(f"Error loading spacy model: {e}")
    print("Ensure en_core_web_sm is installed: python -m spacy download en_core_web_sm")
    exit(1)

if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add custom labels
for label in ["VALUE", "UNIT", "DIRECTION"]:
    ner.add_label(label)

# Disable other pipelines during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for itn in range(20):  # 20 epochs
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=8)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.2, sgd=optimizer, losses=losses)
        print(f"Iteration {itn + 1}, Losses: {losses}")

# Save model
output_dir = Path("./fine_tuned_spacy_ner")
output_dir.mkdir(exist_ok=True)
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# Evaluate on test data
def evaluate_ner(nlp, test_data):
    correct = 0
    total = 0
    for text, annotations in test_data:
        doc = nlp(text)
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        true_entities = annotations["entities"]
        correct += len(set(predicted_entities) & set(true_entities))
        total += len(true_entities)
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.2f}")

evaluate_ner(nlp, test_data)