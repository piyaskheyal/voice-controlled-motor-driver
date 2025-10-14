from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from extract_entities import extract_entities  # Import your NER-based extract_entities
import csv

# Load NER model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_ner_model')
model = AutoModelForTokenClassification.from_pretrained('./fine_tuned_ner_model').to('cuda')

# Initialize NER pipeline
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")

# Define test cases with expected entities
test_cases = [
    {
        "sentence": "Increase the speed by 10 percent",
        "intent": "increase",
        "expected": {'value': 10, 'unit': 'percent', 'direction': None}
    },
    {
        "sentence": "Speed up a little",
        "intent": "increase",
        "expected": {'value': 10, 'unit': 'default', 'direction': None}
    },
    {
        "sentence": "Go full throttle",
        "intent": "increase",
        "expected": {'value': None, 'unit': 'max', 'direction': None}
    },
    {
        "sentence": "Decrease the speed by 25%",
        "intent": "decrease",
        "expected": {'value': 25, 'unit': 'percent', 'direction': None}
    },
    {
        "sentence": "Make it half as fast",
        "intent": "decrease",
        "expected": {'value': None, 'unit': 'half', 'direction': None}
    },
    {
        "sentence": "Stop the motor",
        "intent": "stop",
        "expected": {'value': None, 'unit': None, 'direction': None}
    },
    {
        "sentence": "Change rotation to clockwise",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'clc'}
    },
    {
        "sentence": "Rotate anticlockwise",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'anticlc'}
    },
    {
        "sentence": "Reverse the direction",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'reverse'}
    },
    {
        "sentence": "Change direction",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'reverse'}
    },
    {
        "sentence": "Switch to ccw",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'anticlc'}
    },
    {
        "sentence": "Go cw",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'clc'}
    },
    {
        "sentence": "Flip rotation",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'reverse'}
    },
    {
        "sentence": "Increase by twenty five percent",
        "intent": "increase",
        "expected": {'value': 25, 'unit': 'percent', 'direction': None}
    },
    {
        "sentence": "Rotate anti clockwise",
        "intent": "change_direction",
        "expected": {'value': None, 'unit': None, 'direction': 'anticlc'}
    }
]

def test_ner_model():
    print("Testing NER model...")
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        sentence = case["sentence"]
        intent = case["intent"]
        expected = case["expected"]
        
        # Run extract_entities (uses NER model)
        result = extract_entities(sentence, intent)
        
        # Compare
        print(f"\nSentence: {sentence}")
        print(f"Intent: {intent}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        
        # Check if prediction matches expected
        if result == expected:
            print("Status: PASS")
            correct += 1
        else:
            print("Status: FAIL")
    
    accuracy = (correct / total) * 100
    print(f"\nTest Summary: {correct}/{total} passed ({accuracy:.2f}%)")

def interactive_test():
    print("\nInteractive Testing (type 'exit' to quit)")
    while True:
        sentence = input("Enter command: ")
        if sentence.lower() == 'exit':
            break
        intent = input("Enter intent (increase/decrease/stop/change_direction): ")
        result = extract_entities(sentence, intent)
        print(f"Result: {result}")

if __name__ == "__main__":
    # Run predefined test cases
    test_ner_model()
    
    # Run interactive testing
    interactive_test()