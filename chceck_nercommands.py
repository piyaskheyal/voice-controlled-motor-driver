import csv

with open('ner_commands.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    for i, row in enumerate(reader, 1):
        sentence, labels = row
        sentence_tokens = sentence.split()
        label_tokens = labels.split()
        if len(sentence_tokens) != len(label_tokens):
            print(f"Row {i}: Mismatch in '{sentence}'")
            print(f"Sentence tokens ({len(sentence_tokens)}): {sentence_tokens}")
            print(f"Labels ({len(label_tokens)}): {label_tokens}")