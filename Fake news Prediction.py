import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('C:/Users/alekh/Desktop/news.csv')

# Display the first few rows of the dataset
print(df.head())

# Preprocess the data
X = df['text']  # Update column name
y = df['label']  # Update column name

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Sample prediction
sample_text = ["This is a sample news article to test the model."]
sample_text_tfidf = vectorizer.transform(sample_text)
sample_prediction = model.predict(sample_text_tfidf)
print(f'Sample Prediction: {sample_prediction[0]}')

