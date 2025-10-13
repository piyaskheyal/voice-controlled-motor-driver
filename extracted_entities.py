import spacy
from transformers import pipeline

nlp = spacy.load('en_core_web_sm')
ner = pipeline('ner', model='dslim/bert-base-NER', tokenizer='dslim/bert-base-NER', device=0)  # GPU

def extract_entities(text, intent):
    doc = nlp(text.lower())
    entities = ner(text)
    
    value = None
    unit = 'percent'  # Default
    direction = None
    
    # Extract number (value)
    for ent in doc:
        if ent.like_num:
            value = float(ent.text) if '.' in ent.text else int(ent.text)
            break
    if not value:
        for e in entities:
            if e['entity'].startswith('B') and e['score'] > 0.5:  # NER fallback
                try:
                    value = int(e['word'])
                except:
                    pass
    
    # Units
    if 'percent' in text or '%' in text:
        unit = 'percent'
    elif 'rpm' in text:
        unit = 'rpm'
    elif 'half' in text:
        unit = 'half'
    elif 'quarter' in text:
        unit = 'quarter'
    elif 'double' in text or 'twice' in text:
        unit = 'double'
    elif 'maximum' in text or 'full' in text:
        unit = 'max'
    elif 'minimum' in text or 'idle' in text:
        unit = 'min'
    
    # Directions (for change_direction intent)
    if 'clockwise' in text or 'forward' in text:
        direction = 'clc'
    elif 'counterclockwise' in text or 'anticlockwise' in text or 'anti-clockwise' in text or 'backward' in text or 'reverse' in text:
        direction = 'anticlc'
    elif 'opposite' in text or 'other way' in text or 'flip' in text or 'switch' in text:
        direction = 'reverse'  # You'll handle state flip
    
    if intent in ['increase', 'decrease'] and value is None:
        value = 10  # Default
    
    return {'value': value, 'unit': unit, 'direction': direction}