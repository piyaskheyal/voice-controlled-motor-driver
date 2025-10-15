# Voice-Controlled Motor Driver

A Python-based system for controlling a motor via voice commands, using NLP for intent recognition and entity extraction, and communicating with an ESP32 microcontroller over serial. The ESP32 receives speed and direction commands and displays them in the Serial Monitor.

---

## Features

- **Voice-to-Text**: Converts spoken commands to text.
- **Intent Recognition**: Uses a fine-tuned DistilBERT model to classify motor control intents.
- **Entity Extraction**: Extracts values (speed, direction, units) from commands.
- **Serial Communication**: Sends commands to ESP32 for motor control.
- **ESP32 Integration**: Receives and displays commands for testing.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/voice-controlled-motor-driver.git
cd voice-controlled-motor-driver
```

### 2. Install Python Dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ESP32 Setup

### 1. Serial Port Selection

- Default port is `/dev/ttyACM0` (Linux).
- To change, edit the `port` variable in `control_motor.py`:

```python
# filepath: control_motor.py
port = '/dev/ttyACM0'  # Change to your ESP32 serial port
```

- Find your ESP32 port with:

```bash
ls /dev/ttyACM*
ls /dev/ttyUSB*
```

### 2. ESP32 Test Code

Upload the following sketch to your ESP32 using Arduino IDE or PlatformIO:

```cpp
void setup() {
  Serial.begin(9600);
  while (!Serial) { ; }
  Serial.println("ESP32 Serial Monitor Ready");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      Serial.print("Received command: ");
      Serial.println(command);
    }
  }
}
```

---

## Usage

### 1. Run the Main Program

```bash
python control_motor.py
```

- Speak your command or type it when prompted.
- Supported commands: increase speed, decrease speed, stop, set speed, change direction.
- Example: "Increase speed by 20 percent", "Set speed to 500 rpm", "Change direction to anticlc".

### 2. ESP32 Serial Monitor

- Open the Serial Monitor in Arduino IDE or use `screen`:

```bash
screen /dev/ttyACM0 9600
```

- You should see received commands printed.

---

## Training & Testing the Intent Model

### 1. Training

- Training scripts and data should be placed in the `training/` directory (create if not present).
- Use your own labeled data for motor control intents.
- Example command to train (adjust as needed):

```bash
python training/train_intent_model.py --data training/intent_data.csv --output ./fine_tuned_intent_model
```

### 2. Testing

- Test the model with:

```bash
python training/test_intent_model.py --model ./fine_tuned_intent_model --data training/test_data.csv
```

---

## Adding More Data

- **Intent Data**: Add new labeled examples to `training/intent_data.csv`.
    - Format: `text,intent_label`
- **Entity Extraction**: Update entity patterns in `extract_entities.py` if new units or directions are needed.

---

## File Overview

- `control_motor.py`: Main pipeline for voice-to-motor control.
- `voice_to_text.py`: Converts speech to text.
- `extract_entities.py`: Extracts values and directions from text.
- `requirements.txt`: Python dependencies.
- `training/`: (Create this folder) Scripts and data for model training/testing.

---

## Troubleshooting

- **Serial Port Issues**: Ensure correct port and permissions (`sudo chmod 666 /dev/ttyACM0`).
- **Microphone Issues**: Check `PyAudio` installation and microphone settings.
- **Model Loading Errors**: Ensure `fine_tuned_intent_model` exists and is accessible.

---

## Contributing

- Fork and submit pull requests.
- Add new intents or entity types by updating training data and extraction logic.

---

## License

MIT License

---

## Contact

For questions or issues, open an issue on GitHub or email [your.email@example.com](mailto:your.email@example.com).