import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

user_input = input("Enter a sentence: ").strip() 
user_input = user_input.lower()

tokened_words = word_tokenize(user_input)
print("Tokenized Words:", tokened_words)

stop_words = stopwords.words('english') 
filtered_words = [word for word in tokened_words if word not in stop_words]
print("Filtered Words (No Stopwords):", filtered_words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words]
print("Lemmatized Words:", lemmatized_words)
