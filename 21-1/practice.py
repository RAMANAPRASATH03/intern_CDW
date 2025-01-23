import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
df = pd.read_csv("C:/Users/Kavinprasath/Downloads/all_kindle_review.csv")

# Preprocess the text
stop_words = set(stopwords.words('english'))

# #lemmatize the words given
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# review = [lemmatizer.lemmatize(word) for word in review]
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
wordnet=WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

df['reviewText'] = df['reviewText'].apply(preprocess_text)

# Create Sentiment column (1: Positive, 0: Negative, Drop Neutral)
df['Sentiment'] = df['rating'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else None))
df.dropna(subset=['Sentiment'], inplace=True)

# Prepare the Word2Vec model
corpus = df['reviewText'].tolist()
tokenized_corpus = [word_tokenize(sentence) for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)

# Generate feature vectors for each sentence
def get_feature_vector(sentence, model):
    words = word_tokenize(sentence)
    feature_vector = [model.wv[word] for word in words if word in model.wv]
    if feature_vector:
        return np.mean(feature_vector, axis=0)  # Average of word vectors
    else:
        return np.zeros(model.vector_size)  # Return zero vector if no words match

X = np.array([get_feature_vector(sentence, model) for sentence in corpus])
y = df['Sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
