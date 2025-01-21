import gensim
from gensim.models import word2vec, keyedvectors
import gensim.downloader as api
wv=api.load('word2vec-google-news-300')