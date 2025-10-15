import spacy

# Load fine-tuned model
nlp = spacy.load("./fine_tuned_spacy_ner")

def extract_entities_from_spacy(text):
    doc = nlp(text)
    entities = {'value': None, 'unit': None, 'direction': None}
    
    for ent in doc.ents:
        if ent.label_ == "VALUE":
            try:
                entities['value'] = float(ent.text) if '.' in ent.text else int(ent.text)
            except ValueError:
                entities['value'] = None
        elif ent.label_ == "UNIT":
            unit = ent.text.lower()
            if unit in ['percent', '%']:
                entities['unit'] = '%'
            elif unit in ['rpm']:
                entities['unit'] = 'rpm'
            elif unit in ['half']:
                entities['unit'] = 'half'
            elif unit in ['quarter']:
                entities['unit'] = 'quarter'
            elif unit in ['double']:
                entities['unit'] = 'double'
            elif unit in ['max']:
                entities['unit'] = 'max'
            elif unit in ['min']:
                entities['unit'] = 'min'
        elif ent.label_ == "DIRECTION":
            direction = ent.text.lower()
            if direction in ['clockwise', 'clc']:
                entities['direction'] = 'clc'
            elif direction in ['anticlockwise', 'anticlc', 'counterclockwise']:
                entities['direction'] = 'anticlc'
            elif direction in ['reverse']:
                entities['direction'] = 'reverse'
    
    return entities

if __name__ == "__main__":
    print("SpaCy NER Model Tester (type 'exit' to quit)")
    while True:
        text = input("Enter command: ")
        if text.lower() == 'exit':
            break
        entities = extract_entities_from_spacy(text)
        print(f"Extracted Entities: {entities}")