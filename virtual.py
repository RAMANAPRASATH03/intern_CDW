import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize,word_tokenize
mytext= """In the previous chapter, we saw examples of some common NLP 
applications that we might encounter in everyday life. If we were asked to 
build such an application, think about how we would approach doing so at our 
organization. We would normally walk through the requirements and break the 
problem down into several sub-problems, then try to develop a step-by-step 
procedure to solve them. Since language processing is involved, we would also
list all the forms of text processing."""
mytext=mytext.lower()
sentences=sent_tokenize(mytext)
print("content",sentences)
words=word_tokenize(mytext)
print("content",words)
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')

# init lemmatizer
lemmatizer = WordNetLemmatizer()
words=["trouble","troubling","troubled","troubles","dogs", "cats"]
lemmatized_words=[lemmatizer.lemmatize(word=word,pos='v') for word in words]
print(words, lemmatized_words)

# init stemmer
porter_stemmer=PorterStemmer()
# stem connect variations
words=["connect","connected","connection","connections","connects"]
stemmed_words=[porter_stemmer.stem(word=word) for word in words]
stemdf= pd.DataFrame({'original_word': words,'stemmed_word': stemmed_words})
print(words, stemmed_words)