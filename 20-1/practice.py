import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# User Input
user_input = input("Enter a sentence: ").strip()  # Get input from the user

# Preprocessing Steps
# 1. Convert to lowercase
user_input = user_input.lower()

# 2. Tokenize
tokened_words = word_tokenize(user_input)
print("Tokenized Words:", tokened_words)

# 3. Remove Stopwords
stop_words = stopwords.words('english')  # Load English stopwords
filtered_words = [word for word in tokened_words if word not in stop_words]
print("Filtered Words (No Stopwords):", filtered_words)

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words]
print("Lemmatized Words:", lemmatized_words)
