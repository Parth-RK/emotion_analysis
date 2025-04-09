# Standard library imports
import os
import argparse
from tabulate import tabulate

# Third-party imports
import torch
from transformers import AutoTokenizer

# Local application imports
import config
from model import EmotionClassifier
from preprocess import clean_text
from predict import predict_emotion

def main():
    parser = argparse.ArgumentParser(description="Test emotion prediction model on multiple sentences.")
    parser.add_argument("--model_path", type=str, default=config.BEST_MODEL_PATH, 
                        help="Path to the trained model file (.bin).")
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first.")
        return

    # Set up device, tokenizer and model
    print(f"Using device: {config.DEVICE}")
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print(f"Loading model structure: {config.MODEL_NAME} with {config.NUM_CLASSES} classes")
    model = EmotionClassifier(n_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    try:
        print(f"Loading model weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(config.DEVICE)))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # Define test sentences with expected emotions
    test_sentences = [
        {"text": "How dare they?! The nerve of some people just makes my blood boil; I swear I could just scream.", 
         "expected": "anger"},
        {"text": "Is anything ever going to happen today? Staring at this wall is more interesting than anything else right now.", 
         "expected": "boredom"},
        {"text": "I reach inside for something, anything, but there's just... nothing there anymore. It's like I'm a shell.", 
         "expected": "empty"},
        {"text": "Yes! This is exactly what I wanted to do; I can't wait to dive in and get going!", 
         "expected": "enthusiasm"},
        {"text": "Wow, I haven't laughed this hard in ages! Let's do that again!", 
         "expected": "fun"},
        {"text": "Everything just feels so right, so warm inside; I wouldn't change a single thing about this moment.", 
         "expected": "happiness"},
        {"text": "I can't stand the sight of them; everything they do just grates on my very soul.", 
         "expected": "hate"},
        {"text": "Just being near you makes everything better; my world feels complete when you're here.", 
         "expected": "love"},
        {"text": "Well, that happened. It doesn't really affect me one way or the other, I suppose.", 
         "expected": "neutral"},
        {"text": "Oh, thank goodness that's over. I can finally breathe properly again.", 
         "expected": "relief"},
        {"text": "This ache in my chest just won't go away. It feels like it might rain inside me forever.", 
         "expected": "sadness"},
        {"text": "Whoa, where did that come from?! I absolutely did not see that coming!", 
         "expected": "surprise"},
        {"text": "What if it goes wrong? My stomach is just churning thinking about all the bad things that could happen.", 
         "expected": "worry"},
    ]

    # Run predictions on all test sentences
    results = []
    correct_predictions = 0

    print("\n--- Running predictions on test sentences ---\n")
    
    for item in test_sentences:
        predicted_emotion, probability = predict_emotion(
            item["text"],
            model,
            tokenizer,
            config.DEVICE,
            config.MAX_LEN
        )
        
        # Check if prediction matches expected emotion
        is_correct = predicted_emotion.lower() == item["expected"].lower()
        if is_correct:
            correct_predictions += 1
            
        # Store results
        results.append([
            item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"],
            item["expected"],
            predicted_emotion,
            f"{probability:.4f}",
            "✓" if is_correct else "✗"
        ])

    # Calculate accuracy
    accuracy = correct_predictions / len(test_sentences) * 100

    # Display results in a table
    headers = ["Text", "Expected", "Predicted", "Confidence", "Correct"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Print summary
    print(f"\nAccuracy: {correct_predictions}/{len(test_sentences)} correct predictions ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()