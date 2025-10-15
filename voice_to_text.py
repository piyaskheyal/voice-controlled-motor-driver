import speech_recognition as sr
import logging

# Setup logging
logging.basicConfig(
    filename='voice_control.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def voice_to_text():
    """Capture a single spoken sentence and convert to text."""
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Set up microphone
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            logging.info("Adjusting for ambient noise...")
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            # Prompt user to speak
            print("Speak your command (pause to stop recording):")
            logging.info("Listening for voice command...")
            
            # Listen with pause detection
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Transcribe audio
            try:
                logging.info("Transcribing audio...")
                text = recognizer.recognize_google(audio)
                logging.info(f"Transcribed text: '{text}'")
                print(f"Transcribed text: {text}")
                return text
            except sr.UnknownValueError:
                logging.error("Could not understand audio")
                print("Error: Could not understand audio")
                return None
            except sr.RequestError as e:
                logging.error(f"Speech recognition API error: {e}")
                print(f"Error: Speech recognition API failed: {e}")
                return None
            except Exception as e:
                logging.error(f"Unexpected error during transcription: {e}")
                print(f"Error: Unexpected transcription error: {e}")
                return None
    except Exception as e:
        logging.error(f"Microphone error: {e}")
        print(f"Error: Microphone setup failed: {e}")
        return None

if __name__ == "__main__":
    print("Voice-to-Text Converter (type 'exit' to quit)")
    while True:
        command = input("Press Enter to speak or type 'exit' to quit: ")
        if command.lower() == 'exit':
            logging.info("Exiting voice-to-text")
            break
        text = voice_to_text()
        if text:
            print(f"Command: {text}")
        else:
            print("No command detected. Try again.")