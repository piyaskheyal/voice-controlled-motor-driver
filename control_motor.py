import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from extract_entities import extract_entities
import serial
import time

# Load intent model and tokenizer (offline)
model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_intent_model').to('cuda')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
label_map = {0: "increase", 1: "decrease", 2: "stop", 3: "change_direction"}

# Motor state
current_speed = 0  # PWM 0-255
current_direction = 'clc'  # Default: clockwise
MAX_PWM = 255
MAX_RPM = 1000  # Adjust based on motor specs (e.g., 1000 RPM = max PWM)

def predict_intent(text):
    """Predict intent using fine-tuned model."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32).to('cuda')
    outputs = model(**inputs)
    intent_id = torch.argmax(outputs.logits, dim=1).item()
    return label_map[intent_id]

def map_to_command(intent, entities, current_speed, current_direction):
    """
    Map intent and entities to ESP32 command (speed 0-255, direction clc/anticlc).
    Returns: (new_speed, new_direction)
    """
    value = entities['value']
    unit = entities['unit']
    direction = entities['direction']
    
    new_speed = current_speed
    new_direction = current_direction
    
    if intent in ['increase', 'decrease']:
        # Calculate delta
        if unit == 'percent':
            delta = int(MAX_PWM * (value / 100)) if value else 0
        elif unit == 'rpm':
            delta = int(value / MAX_RPM * MAX_PWM)  # Scale RPM to PWM
        elif unit == 'half':
            delta = current_speed // 2
        elif unit == 'quarter':
            delta = current_speed // 4
        elif unit == 'double':
            delta = current_speed
        elif unit == 'max':
            delta = MAX_PWM - current_speed
        elif unit == 'min':
            delta = current_speed  # To 0
        elif unit == 'default':
            delta = int(MAX_PWM * (10 / 100))  # Default 10%
        else:
            delta = 0
        
        # Apply delta
        if intent == 'increase':
            new_speed = min(current_speed + delta, MAX_PWM)
        else:  # decrease
            new_speed = max(current_speed - delta, 0)
    
    elif intent == 'stop':
        new_speed = 0
    
    elif intent == 'change_direction':
        if direction == 'reverse':
            new_direction = 'anticlc' if current_direction == 'clc' else 'clc'
        else:
            new_direction = direction  # clc or anticlc
    
    return new_speed, new_direction

def send_to_esp32(speed, direction):
    """Send command to ESP32 via serial."""
    try:
        with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:  # Adjust port/baudrate
            command = f"{speed},{direction}\n".encode()
            ser.write(command)
            time.sleep(0.1)  # Small delay for serial stability
    except serial.SerialException as e:
        print(f"Serial error: {e}")

def process_command(text):
    """Full pipeline: Predict intent, extract entities, map to command, send to ESP32."""
    global current_speed, current_direction
    
    # Predict intent
    intent = predict_intent(text)
    
    # Extract entities
    entities = extract_entities(text, intent)
    
    # Map to command
    new_speed, new_direction = map_to_command(intent, entities, current_speed, current_direction)
    
    # Update state
    current_speed = new_speed
    current_direction = new_direction
    
    # Send to ESP32
    send_to_esp32(new_speed, new_direction)
    
    return {"intent": intent, "entities": entities, "speed": new_speed, "direction": new_direction}

# Test function
if __name__ == "__main__":
    # test_cases = [
    #     "Increase the speed by 20 percent",
    #     "Speed up a little",
    #     "Go full throttle",
    #     "Decrease the speed by 25%",
    #     "Make it half as fast",
    #     "Stop the motor",
    #     "Change rotation to clockwise",
    #     "Reverse the direction"
    # ]
    
    # for text in test_cases:
    #     result = process_command(text)
    #     print(f"Text: {text} -> {result}")
    while True:
        print(process_command(input("Enter command: ")))