
# %% [markdown]
# ## pretrained

# %%
import gensim

# %%
from gensim.models import Word2Vec, KeyedVectors

# %%
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

# %%
vec_king = wv['king']
vec_king

# %%
vec_king.shape

# %%
wv['cricket']

# %%
wv.most_similar('cricket')

# %%
wv.most_similar('happy')

# %%
wv.similarity("hockey","sports")

# %%
vec=wv['king']-wv['man']+wv['women']

# %%
vec

# %%
wv.most_similar([vec])

# %% [markdown]
# ## train our own model

# %%
# Importing necessary libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec, KeyedVectors

# %%
# Ensure you have the necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Messy data with numbers, random characters, and typos
messages = pd.DataFrame({
    'message': [
        "Why don't skeletons fight each other? They don't have the guts! #funny",
        "I told my computer I needed a break... and now it won't stop sending me Kit-Kats! #lol",
        "I c0uldn't figure out h0w to put my seatbelt on, but th3n it clicked!",
        "I used to play piano by ear!! But now I use my hands! #funny",
        "I'm reading a book on anti-gravity... it's impossible to put down! #gravity",
        "Why don't programmers like nature? It has too many bugs!! #programming #coding",
        "I told my wife she was drawing her eyebrows too high! She looked surprised! 1234"
    ]
})

# %%
# Initialize the Lemmatizer
lemmatizer = WordNetLemmatizer()

# %%

# Corpus list to hold cleaned sentences
corpus = []

# Process each message in the DataFrame
for i in range(0, len(messages)):
    # Remove non-alphabetic characters, digits, and extra spaces
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    
    # Convert to lowercase
    review = review.lower()
    
    # Tokenize the sentence into words
    review = review.split()
    
    # Lemmatize each word
    review = [lemmatizer.lemmatize(word) for word in review]
    
    # Rejoin words to form the cleaned sentence
    review = ' '.join(review)
    
    # Append to the corpus
    corpus.append(review)

# Print the cleaned corpus
print("Cleaned Corpus:")
print(corpus)

# %%
# Tokenize the cleaned corpus for Word2Vec model training
tokenized_corpus = [word_tokenize(sentence) for sentence in corpus]

# %%
tokenized_corpus

# %%
# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0)

# Save the model (optional)
model.save("messy_word2vec.model")

# %%
vec_computer = model.wv['computer']
vec_computer

# %%
vec_computer.shape

# %%
# Example: Get the vector representation of a word ('computer' in this case)
# Since our corpus doesn't include "computer", let's use a word from our corpus like "computer"
word_vector = model.wv['computer'] if 'computer' in model.wv else None
print(f"\nVector for 'computer': {word_vector}")

# %%
# Example: Find words similar to 'computer'
similar_words = model.wv.most_similar('computer', topn=5) if 'computer' in model.wv else None
print(f"\nWords similar to 'computer': {similar_words}")

# %%
# Check similarity between two words ('computer' and 'programmer')
similarity = model.wv.similarity('computer', 'programmer') if 'computer' in model.wv and 'programmer' in model.wv else None
if similarity:
    print(f"\nSimilarity between 'computer' and 'programmer': {similarity}")
else:
    print("\nSimilarity could not be calculated between 'computer' and 'programmer'.")

# %%



