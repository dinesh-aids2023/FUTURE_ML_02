import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# Load dataset
df = pd.read_csv("/content/IMDB-Dataset.csv") 
df.info() 
# Check the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Check class distribution
df['sentiment'].value_counts()
# Function to clean text
import string

def clean_text(text):
    text = text.lower() 
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()  
    return ' '.join(words)

# Apply the cleaning function to the review column
df['cleaned_review'] = df['review'].apply(clean_text)

# Check the cleaned text
df[['review', 'cleaned_review']].head()
X = df['cleaned_review']  
y = df['sentiment']  

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert text data into numerical data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict the sentiment for the test set
y_pred = model.predict(X_test_tfidf)

# Check accuracy and print classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix to see how well the model is performing
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Example review
new_review = input("enter your review")

# Clean and vectorize the new review
new_review_cleaned = clean_text(new_review)
new_review_tfidf = vectorizer.transform([new_review_cleaned])

# Predict the sentiment
predicted_sentiment = model.predict(new_review_tfidf)
print(f"Predicted Sentiment: {predicted_sentiment[0]}")
