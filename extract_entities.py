from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load NER model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_ner_model')
model = AutoModelForTokenClassification.from_pretrained('./fine_tuned_ner_model').to('cuda')

# Initialize NER pipeline
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")

def extract_entities(text, intent):
    """
    Extract entities using fine-tuned NER model.
    
    Args:
        text (str): Input command (e.g., "Increase speed by 10 percent").
        intent (str): Predicted intent (increase, decrease, stop, change_direction).
    
    Returns:
        dict: {'value': float or int, 'unit': str, 'direction': str or None}
    """
    entities = ner_pipeline(text.lower())
    
    value = None
    unit = None
    direction = None
    
    # Parse NER results
    for ent in entities:
        if ent['entity_group'].startswith('VALUE'):
            try:
                value = float(ent['word']) if '.' in ent['word'] else int(ent['word'])
            except:
                pass
        elif ent['entity_group'].startswith('UNIT'):
            unit = ent['word']
        elif ent['entity_group'].startswith('DIRECTION'):
            if any(x in ent['word'] for x in ['clockwise', 'forward', 'cw']):
                direction = 'clc'
            elif any(x in ent['word'] for x in ['counterclockwise', 'anticlockwise', 'anti clockwise', 'backward', 'ccw']):
                direction = 'anticlc'
            else:
                direction = 'reverse'
    
    # Defaults for increase/decrease
    if intent in ['increase', 'decrease'] and value is None:
        value = 10
        unit = 'default' if unit is None else unit
    
    # Default for ambiguous change_direction
    if intent == 'change_direction' and direction is None:
        direction = 'reverse'
    
    return {'value': value, 'unit': unit, 'direction': direction}

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
        ("Reverse the direction", "change_direction"),
        ("Change direction", "change_direction")
    ]
    
    for text, intent in test_cases:
        result = extract_entities(text, intent)
        print(f"Text: {text} (Intent: {intent}) -> {result}")