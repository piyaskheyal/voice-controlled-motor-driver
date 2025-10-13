max_speed = 255

def process_command(intent, entities, current_speed, current_dir):
    value = entities['value']
    unit = entities['unit']
    
    if unit == 'percent':
        delta = int(max_speed * (value / 100))
    elif unit == 'rpm':  # Assume RPM maps to PWM; adjust formula
        delta = value  # Or scale: int(value / some_rpm_max * max_speed)
    elif unit == 'half':
        delta = current_speed // 2
    elif unit == 'quarter':
        delta = current_speed // 4
    elif unit == 'double':
        delta = current_speed
    elif unit == 'max':
        delta = max_speed - current_speed
    elif unit == 'min':
        delta = current_speed
    
    if intent == 'increase':
        new_speed = min(current_speed + delta, max_speed)
    elif intent == 'decrease':
        new_speed = max(current_speed - delta, 0)
    elif intent == 'stop':
        new_speed = 0
    elif intent == 'change_direction':
        if entities['direction'] == 'reverse':
            new_dir = 'anticlc' if current_dir == 'clc' else 'clc'
        else:
            new_dir = entities['direction']
        return new_speed, new_dir  # Send to ESP32
    
    return new_speed, current_dir