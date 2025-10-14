import torch
import serial
import time
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from extract_entities import extract_entities

# Setup logging
logging.basicConfig(
    filename='motor_control.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load intent model and tokenizer (offline)
try:
    model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_intent_model').to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
except Exception as e:
    logging.error(f"Failed to load intent model: {e}")
    print(f"Error: Could not load intent model: {e}")
    exit(1)

# Intent label mapping
label_map = {0: "increase", 1: "decrease", 2: "stop", 3: "change_direction"}

# Motor state
current_speed = 0  # PWM 0-255
current_direction = 'clc'  # Default: clockwise
MAX_PWM = 255
MAX_RPM = 1000  # Adjust based on motor specs

def predict_intent(text):
    """Predict intent using fine-tuned DistilBERT model."""
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=32).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
        intent_id = torch.argmax(outputs.logits, dim=1).item()
        intent = label_map[intent_id]
        logging.info(f"Predicted intent for '{text}': {intent}")
        return intent
    except Exception as e:
        logging.error(f"Intent prediction failed for '{text}': {e}")
        print(f"Error: Intent prediction failed: {e}")
        return None

def map_to_command(intent, entities, current_speed, current_direction):
    """
    Map intent and entities to ESP32 command (speed 0-255, direction clc/anticlc).
    Returns: (new_speed, new_direction)
    """
    if not intent:
        return current_speed, current_direction
    
    value = entities.get('value')
    unit = entities.get('unit')
    direction = entities.get('direction')
    
    new_speed = current_speed
    new_direction = current_direction
    
    # Validate entities
    if value is not None and not isinstance(value, (int, float)):
        logging.warning(f"Invalid value '{value}' for intent '{intent}'")
        value = None
    if direction not in [None, 'clc', 'anticlc', 'reverse']:
        logging.warning(f"Invalid direction '{direction}' for intent '{intent}'")
        direction = None
    
    if intent in ['increase', 'decrease']:
        # Calculate delta
        delta = 0
        if unit == 'percent' and value is not None:
            delta = int(MAX_PWM * (value / 100))
        elif unit == 'rpm' and value is not None:
            delta = int(value / MAX_RPM * MAX_PWM)
        elif unit == 'half':
            delta = current_speed // 2
        elif unit == 'quarter':
            delta = current_speed // 4
        elif unit == 'double':
            delta = current_speed
        elif unit == 'max':
            delta = MAX_PWM - current_speed
        elif unit == 'min':
            delta = current_speed
        elif unit == 'default':
            delta = int(MAX_PWM * (10 / 100))  # Default 10%
        
        # Apply delta
        if intent == 'increase':
            new_speed = min(current_speed + delta, MAX_PWM)
        else:  # decrease
            new_speed = max(current_speed - delta, 0)
    
    elif intent == 'stop':
        new_speed = 0
    
    elif intent == 'change_direction':
        if direction == 'clc':
            new_direction = 'clc'
        elif direction == 'anticlc':
            new_direction = 'anticlc'
        elif direction == 'reverse':
            new_direction = 'anticlc' if current_direction == 'clc' else 'clc'
        else:
            logging.warning(f"No valid direction for 'change_direction' in '{intent}'")
            new_direction = 'anticlc' if current_direction == 'clc' else 'clc'
    
    logging.info(f"Mapped to command: speed={new_speed}, direction={new_direction}")
    return new_speed, new_direction

def send_to_esp32(speed, direction):
    """Send command to ESP32 via serial."""
    port = '/dev/ttyACM0'  # Manually set port
    try:
        with serial.Serial(port, 115200, timeout=1) as ser:
            ser.reset_input_buffer()  # Clear any stale data
            command = f"{speed},{direction}\n".encode()
            ser.write(command)
            time.sleep(0.1)  # Serial stability
            logging.info(f"Sent to ESP32 on {port}: {command.decode().strip()}")
            print(f"Motor set to speed {speed}, direction {direction}")
            return True
    except serial.SerialException as e:
        logging.error(f"Serial error on {port}: {e}")
        print(f"Error: Could not send to ESP32: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error on {port}: {e}")
        print(f"Error: Unexpected serial error: {e}")
        return False

def process_command(text):
    """Full pipeline: Predict intent, extract entities, map to command, send to ESP32."""
    global current_speed, current_direction
    
    if not text or not text.strip():
        logging.warning("Empty or invalid command received")
        print("Error: Please enter a valid command")
        return None
    
    # Predict intent
    intent = predict_intent(text)
    if not intent:
        return None
    
    # Extract entities
    try:
        entities = extract_entities(text, intent)
    except Exception as e:
        logging.error(f"Entity extraction failed for '{text}': {e}")
        print(f"Error: Entity extraction failed: {e}")
        return None
    
    # Map to command
    new_speed, new_direction = map_to_command(intent, entities, current_speed, current_direction)
    
    # Update state
    current_speed = new_speed
    current_direction = new_direction
    
    # Send to ESP32
    success = send_to_esp32(new_speed, new_direction)
    
    result = {
        "intent": intent,
        "entities": entities,
        "speed": new_speed,
        "direction": new_direction,
        "success": success
    }
    logging.info(f"Processed command '{text}': {result}")
    return result

# Test function
if __name__ == "__main__":
    print("Motor Control (type 'exit' to quit)")
    while True:
        text = input("Enter command: ")
        if text.lower() == 'exit':
            logging.info("Exiting motor control")
            break
        result = process_command(text)
        if result:
            print(f"Result: {result}")
        else:
            print("Command failed. Check logs for details.")