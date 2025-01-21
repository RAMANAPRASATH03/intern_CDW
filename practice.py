import nltk
import pandas as pd
from nltk import word_tokenize, wordpunct_tokenize
words_token="WHAT MADE A MAN TO THINK OF HIS OWN ?"
words_token=words_token.lower()
tokened_words=word_tokenize(words_token)
print(tokened_words)

from nltk import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
words=["running","runs"]
lemmatized_words=[lemmatizer.lemmatize(word=word,pos='v') for word in words]
print(words,lemmatized_words)

from nltk.corpus import stopwords
nltk.download('stopwords')
print(stopwords.words_token)