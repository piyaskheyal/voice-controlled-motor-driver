from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import logging

# Setup logging
logging.basicConfig(
    filename='extract_entities.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load NER model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_ner_model')
    model = AutoModelForTokenClassification.from_pretrained('./fine_tuned_ner_model').to('cuda')
except Exception as e:
    logging.error(f"Failed to load NER model: {e}")
    raise

# Initialize NER pipeline
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")

def extract_entities(text, intent):
    """
    Extract entities using fine-tuned NER model, then refine direction with rule-based logic (regex).
    
    Args:
        text (str): Input command (e.g., "Rotate anticlockwise").
        intent (str): Predicted intent (increase, decrease, stop, change_direction).
    
    Returns:
        dict: {'value': float or int, 'unit': str, 'direction': str or None}
    """
    logging.info(f"Processing text: '{text}' with intent: {intent}")
    
    # Run NER
    entities = ner_pipeline(text.lower())
    logging.info(f"NER output: {entities}")
    
    value = None
    unit = None
    direction = None
    direction_str = None
    
    # Parse NER results
    for ent in entities:
        if ent['entity_group'].startswith('VALUE'):
            try:
                value = float(ent['word']) if '.' in ent['word'] else int(ent['word'])
                logging.info(f"Extracted value: {value}")
            except:
                logging.warning(f"Failed to parse value: {ent['word']}")
        elif ent['entity_group'].startswith('UNIT'):
            unit = ent['word']
            logging.info(f"Extracted unit: {unit}")
        elif ent['entity_group'].startswith('DIRECTION'):
            direction_str = ent['word']
            logging.info(f"Extracted direction_str: {direction_str}")
    
    # Refine direction with regex on full text (fallback if NER misses)
    text_lower = text.lower()
    if re.search(r'\b(clockwise|cw|forward)\b', text_lower, re.IGNORECASE):
        direction = 'clc'
        logging.info(f"Matched 'clc' via regex on text: {text_lower}")
    elif re.search(r'\b(counterclockwise|anticlockwise|ccw|anti\s*clockwise|backward)\b', text_lower, re.IGNORECASE):
        direction = 'anticlc'
        logging.info(f"Matched 'anticlc' via regex on text: {text_lower}")
    elif direction_str:
        # Check NER direction_str if present
        if re.search(r'\b(clockwise|cw|forward)\b', direction_str, re.IGNORECASE):
            direction = 'clc'
            logging.info(f"Matched 'clc' via regex on direction_str: {direction_str}")
        elif re.search(r'\b(counterclockwise|anticlockwise|ccw|anti\s*clockwise|backward)\b', direction_str, re.IGNORECASE):
            direction = 'anticlc'
            logging.info(f"Matched 'anticlc' via regex on direction_str: {direction_str}")
        else:
            direction = 'reverse'
            logging.info(f"Defaulted to 'reverse' for direction_str: {direction_str}")
    elif intent == 'change_direction':
        # Default for ambiguous change_direction
        direction = 'reverse'
        logging.info(f"Defaulted to 'reverse' for intent: {intent}, no direction keywords")
    
    # Defaults for increase/decrease
    if intent in ['increase', 'decrease'] and value is None:
        value = 10
        unit = 'default' if unit is None else unit
        logging.info(f"Applied default value: {value}, unit: {unit} for intent: {intent}")
    
    result = {'value': value, 'unit': unit, 'direction': direction}
    logging.info(f"Final entities: {result}")
    return result

# Test function
if __name__ == "__main__":
    test_cases = [
        ("Increase the speed by 10 percent", "increase"),
        ("Speed up a little", "increase"),
        ("Go full throttle", "increase"),
        ("Decrease the speed by 25%", "decrease"),
        ("Make it half as fast", "decrease"),
        ("Stop the motor", "stop"),
        ("Change rotation to clockwise", "change_direction"),
        ("Rotate anticlockwise", "change_direction"),
        ("Rotate anti clockwise", "change_direction"),
        ("Change rotation to counterclockwise", "change_direction"),
        ("Reverse the direction", "change_direction"),
        ("Change direction", "change_direction"),
        ("Switch to ccw", "change_direction"),
        ("Go cw", "change_direction"),
        ("Flip rotation", "change_direction"),
        ("Other way around", "change_direction")
    ]
    
    for text, intent in test_cases:
        result = extract_entities(text, intent)
        print(f"Text: {text} (Intent: {intent}) -> {result}")