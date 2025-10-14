import spacy

# Load SpaCy model (offline, assumes en_core_web_sm is downloaded)
nlp = spacy.load('en_core_web_sm')

def extract_entities(text, intent):
    """
    Extract entities (value, unit, direction) from command text based on intent.
    
    Args:
        text (str): Input command (e.g., "Increase speed by 10 percent").
        intent (str): Predicted intent (increase, decrease, stop, change_direction).
    
    Returns:
        dict: {'value': float or int, 'unit': str, 'direction': str or None}
              - value: Numeric value (e.g., 10, 25, 500) or None.
              - unit: percent, rpm, half, quarter, double, max, min, default, or None.
              - direction: clc, anticlc, reverse, or None.
    """
    doc = nlp(text.lower())
    
    value = None
    unit = 'percent' if intent in ['increase', 'decrease'] else None  # Default for increase/decrease
    direction = None
    
    # Extract number (value)
    for token in doc:
        if token.like_num:
            try:
                value = float(token.text) if '.' in token.text else int(token.text)
                break
            except ValueError:
                pass
    
    # Units (based on command text)
    text_lower = text.lower()
    if any(x in text_lower for x in ['percent', '%']):
        unit = 'percent'
    elif 'rpm' in text_lower:
        unit = 'rpm'
    elif 'half' in text_lower:
        unit = 'half'
    elif 'quarter' in text_lower:
        unit = 'quarter'
    elif any(x in text_lower for x in ['double', 'twice']):
        unit = 'double'
    elif any(x in text_lower for x in ['maximum', 'full', 'throttle']):
        unit = 'max'
    elif any(x in text_lower for x in ['minimum', 'idle']):
        unit = 'min'
    elif any(x in text_lower for x in ['little', 'slightly', 'bit', 'gradually', 'faster', 'slower']):
        unit = 'default'  # For vague terms, use default value
    
    # Directions (for change_direction intent)
    if intent == 'change_direction':
        if any(x in text_lower for x in ['clockwise', 'forward']):
            direction = 'clc'
        elif any(x in text_lower for x in ['counterclockwise', 'anticlockwise', 'anti-clockwise', 'backward']):
            direction = 'anticlc'
        elif any(x in text_lower for x in ['reverse', 'opposite', 'other way', 'flip', 'switch']):
            direction = 'reverse'  # Flip current direction (handled later)
    
    # Apply defaults for increase/decrease
    if intent in ['increase', 'decrease'] and value is None:
        if unit == 'default' or any(x in text_lower for x in ['faster', 'slower', 'quicker', 'more', 'raise', 'boost', 'add', 'ramp', 'step', 'reduce', 'lower', 'ease', 'drop']):
            value = 10  # Default as specified
    
    return {'value': value, 'unit': unit, 'direction': direction}

# Test function
if __name__ == "__main__":
    test_cases = [
        ("Increase the speed by 10 percent", "increase"),
        ("Speed up a little", "increase"),
        ("Go full throttle", "increase"),
        ("Decrease the speed by 25%", "decrease"),
        ("Make it half as fast", "decrease"),
        ("Slow it to minimum", "decrease"),
        ("Stop the motor", "stop"),
        ("Change rotation to clockwise", "change_direction"),
        ("Reverse the direction", "change_direction"),
        ("Switch to anti-clockwise rotation", "change_direction")
    ]
    
    for text, intent in test_cases:
        result = extract_entities(text, intent)
        print(f"Text: {text} (Intent: {intent}) -> {result}")