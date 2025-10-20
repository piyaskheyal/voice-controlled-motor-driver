import spacy
import re
import logging

# Setup logging
logging.basicConfig(
    filename='motor_control.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load fine-tuned spaCy model
try:
    nlp = spacy.load("./fine_tuned_spacy_ner")
except Exception as e:
    logging.error(f"Failed to load spaCy model: {e}")
    raise RuntimeError(f"Failed to load spaCy model: {e}")

def extract_entities(text, intent=None):
    """
    Extract entities (value, unit, direction) from text using spaCy NER model with regex fallback.
    Args:
        text (str): Input command (e.g., "set the speed to 90%").
        intent (str, optional): Intent from predict_intent (e.g., "set_speed").
    Returns:
        dict: {'value': int/float/None, 'unit': str/None, 'direction': str/None}
    """
    if not text or not text.strip():
        logging.warning("Empty or invalid input text for entity extraction")
        return {'value': None, 'unit': None, 'direction': None}

    try:
        # Lowercase for consistency
        text = text.lower()
        doc = nlp(text)
        entities = {'value': None, 'unit': None, 'direction': None}
        
        for ent in doc.ents:
            if ent.label_ == "VALUE":
                try:
                    # Handle percentage or numeric values
                    value_text = ent.text.replace('%', '').strip()
                    entities['value'] = float(value_text) if '.' in value_text else int(value_text)
                except ValueError:
                    logging.warning(f"Invalid VALUE entity: '{ent.text}'")
                    entities['value'] = None
            elif ent.label_ == "UNIT":
                unit = ent.text.lower()
                if unit in ['percent', '%']:
                    entities['unit'] = '%'
                elif unit in ['half']:
                    entities['unit'] = 'half'
                elif unit in ['quarter']:
                    entities['unit'] = 'quarter'
                elif unit in ['double']:
                    entities['unit'] = 'double'
                elif unit in ['max', 'maximum']:
                    entities['unit'] = 'max'
                elif unit in ['min', 'minimum']:
                    entities['unit'] = 'min'
                else:
                    logging.warning(f"Unknown UNIT entity: '{unit}'")
                    entities['unit'] = None
            elif ent.label_ == "DIRECTION":
                direction = ent.text.lower()
                if direction in ['clockwise', 'clc']:
                    entities['direction'] = 'clc'
                elif direction in ['anticlockwise', 'anticlc', 'counterclockwise']:
                    entities['direction'] = 'anticlc'
                elif direction in ['reverse']:
                    entities['direction'] = 'reverse'
                elif direction in ['max', 'maximum', 'min', 'minimum']:
                    logging.warning(f"Misclassified DIRECTION entity: '{direction}', correcting to UNIT")
                    entities['direction'] = None
                    entities['unit'] = 'max' if direction in ['max', 'maximum'] else 'min'
                else:
                    logging.warning(f"Unknown DIRECTION entity: '{direction}'")
                    entities['direction'] = None
        
        # Regex fallback for max/min
        if entities['unit'] is None:
            if re.search(r'\b(max|maximum)\b', text):
                entities['unit'] = 'max'
                logging.info(f"Regex fallback: Set unit to 'max' for '{text}'")
            elif re.search(r'\b(min|minimum)\b', text):
                entities['unit'] = 'min'
                logging.info(f"Regex fallback: Set unit to 'min' for '{text}'")
        
        # Intent-based fallback for direction
        if intent == "change_direction" and entities['direction'] is None:
            entities['direction'] = 'reverse'
            logging.info(f"Fallback: Set direction to 'reverse' for intent 'change_direction'")

        logging.info(f"Extracted entities from '{text}': {entities}")
        return entities
    
    except Exception as e:
        logging.error(f"Entity extraction failed for '{text}': {e}")
        return {'value': None, 'unit': None, 'direction': None}

if __name__ == "__main__":
    print("SpaCy NER Entity Extractor Tester (type 'exit' to quit)")
    while True:
        text = input("Enter command: ")
        if text.lower() == 'exit':
            break
        entities = extract_entities(text)
        print(f"Extracted Entities: {entities}")