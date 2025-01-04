## Run The Model
import joblib

# Load the saved vectorizer and model
vectorizer = joblib.load('tweet_vectorizer.pkl')
classifier = joblib.load('emotion_classifier.pkl')

# Predict emotion with confidence scores
def predict_with_confidence(tweet_texts):
    tweet_vectorized = vectorizer.transform(tweet_texts)
    predicted_emotions = classifier.predict(tweet_vectorized)
    predicted_probabilities = classifier.predict_proba(tweet_vectorized)

    results = []
    for tweet, emotion, probs in zip(tweet_texts, predicted_emotions, predicted_probabilities):
        confidence = max(probs) * 100
        results.append({
            'tweet': tweet,
            'predicted_emotion': emotion,
            'confidence': f"{confidence:.2f}%"
        })
    return results

# Example usage
new_tweets = [
    "Feeling amazing today, everything is going great!",
    "I am so sad and lonely right now.",
    "This is the best day of my life!",
    "I feel neutral about this situation."
]
prediction_results = predict_with_confidence(new_tweets)
for result in prediction_results:
    print(f"Tweet: {result['tweet']}\nPredicted Emotion: {result['predicted_emotion']}\nConfidence: {result['confidence']}\n")