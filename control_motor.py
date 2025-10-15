import torch
import serial
import time
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from extract_entities import extract_entities
from voice_to_text import voice_to_text

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
label_map = {0: "increase", 1: "decrease", 2: "stop", 3: "set_speed", 4: "change_direction"}

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
        logging.warning("No intent provided, returning current state")
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
    
    logging.info(f"Input - intent: {intent}, entities: {entities}, current_speed: {current_speed}, current_direction: {current_direction}")

    if intent in ['increase', 'decrease']:
        # Incremental change based on current_speed
        delta = 0
        if unit == '%' and value is not None:
            delta = int(MAX_PWM * (value / 100))
            logging.info(f"Percent unit detected, value={value}, delta={delta}")
        elif unit == 'rpm' and value is not None:
            delta = int(value / MAX_RPM * MAX_PWM)
            logging.info(f"RPM unit detected, value={value}, delta={delta}")
        elif unit == 'half':
            delta = current_speed // 2
            logging.info(f"Half unit detected, delta={delta}")
        elif unit == 'quarter':
            delta = current_speed // 4
            logging.info(f"Quarter unit detected, delta={delta}")
        elif unit == 'double':
            delta = current_speed
            logging.info(f"Double unit detected, delta={delta}")
        elif unit == 'max':
            delta = MAX_PWM - current_speed
            logging.info(f"Max unit detected, delta={delta}")
        elif unit == 'min':
            delta = current_speed
            logging.info(f"Min unit detected, delta={delta}")
        elif unit == 'default':
            delta = int(MAX_PWM * (10 / 100))  # Default 10%
            logging.info(f"Default unit detected, delta={delta}")
        else:
            logging.warning(f"Unknown unit '{unit}' for intent '{intent}', no change")
        
        # Apply delta
        if intent == 'increase':
            new_speed = min(current_speed + delta, MAX_PWM)
        else:  # decrease
            new_speed = max(current_speed - delta, 0)
    
    elif intent == 'set_speed':
        # Set absolute speed
        if unit == '%' and value is not None:
            new_speed = min(int(MAX_PWM * (value / 100)), MAX_PWM)
            logging.info(f"Set percent speed, value={value}, new_speed={new_speed}")
        elif unit == 'rpm' and value is not None:
            new_speed = min(int(value / MAX_RPM * MAX_PWM), MAX_PWM)
            logging.info(f"Set RPM speed, value={value}, new_speed={new_speed}")
        elif unit == 'half':
            new_speed = MAX_PWM // 2
            logging.info(f"Set half speed, new_speed={new_speed}")
        elif unit == 'quarter':
            new_speed = MAX_PWM // 4
            logging.info(f"Set quarter speed, new_speed={new_speed}")
        elif unit == 'double':
            new_speed = MAX_PWM  # Full speed for double
            logging.info(f"Set double speed, new_speed={new_speed}")
        elif unit == 'max':
            new_speed = MAX_PWM
            logging.info(f"Set max speed, new_speed={new_speed}")
        elif unit == 'min':
            new_speed = 0
            logging.info(f"Set min speed, new_speed={new_speed}")
        elif unit == 'default':
            new_speed = int(MAX_PWM * 0.1)  # Default 10%
            logging.info(f"Set default speed, new_speed={new_speed}")
        else:
            logging.warning(f"Unknown unit '{unit}' for intent '{intent}', no change")
    
    elif intent == 'stop':
        new_speed = 0
        logging.info("Stop command, new_speed=0")
    
    elif intent == 'change_direction':
        if direction == 'clc':
            new_direction = 'clc'
        elif direction == 'anticlc':
            new_direction = 'anticlc'
        elif direction == 'reverse':
            new_direction = 'anticlc' if current_direction == 'clc' else 'clc'
        else:
            logging.warning(f"No valid direction for 'change_direction', toggling direction")
            new_direction = 'anticlc' if current_direction == 'clc' else 'clc'
        logging.info(f"Direction set to: {new_direction}")
    
    logging.info(f"Output - new_speed={new_speed}, new_direction={new_direction}")
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
    
    logging.info(f"Processing command: '{text}', current state: speed={current_speed}, direction={current_direction}")
    
    # Predict intent
    intent = predict_intent(text)
    if not intent:
        return None
    
    # Extract entities
    try:
        entities = extract_entities(text, intent)
        logging.info(f"Extracted entities: {entities}")
    except Exception as e:
        logging.error(f"Entity extraction failed for '{text}': {e}")
        print(f"Error: Entity extraction failed: {e}")
        return None
    
    # Map to command
    new_speed, new_direction = map_to_command(intent, entities, current_speed, current_direction)
    
    # Update state
    logging.info(f"Updating state: current_speed={current_speed} -> {new_speed}, current_direction={current_direction} -> {new_direction}")
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

if __name__ == "__main__":
    print("Voice-Controlled Motor (type 'exit' to quit)")
    while True:
        command = input("Press Enter to speak or type 'exit' to quit: ")
        if command.lower() == 'exit':
            logging.info("Exiting motor control")
            break
        text = voice_to_text()
        if text:
            result = process_command(text)
            if result:
                print(f"Result: {result}")
            else:
                print("Command failed. Check logs for details.")
        else:
            print("No command detected. Try again.")