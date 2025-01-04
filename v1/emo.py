# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
dataset_path = 'emotion_data.csv'
data = pd.read_csv(dataset_path)

# Inspect the data
print(data.head())
print(data.info())

# Rename columns for clarity
data.rename(columns={
    'content': 'tweet_text',
    'sentiment': 'emotion_label',
}, inplace=True)

# Prepare the data for model training
X = data['tweet_text']
y = data['emotion_label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the vectorizer and model for future use
joblib.dump(vectorizer, 'tweet_vectorizer.pkl')
joblib.dump(classifier, 'emotion_classifier.pkl')



