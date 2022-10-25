import regex
from pprint import pprint
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import sklearn
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize,sent_tokenize
import pandas as pd
#nltk.download('punkt')
file = open("data/corpus.txt","r")
corpus=file.read()
sentences=regex.findall("[A-Za-z0-9 '\"]",corpus.lower())
print(sentences)
train, test = train_test_split(sentences, test_size = 0.1)
#print(train)
#train_tokens=regex.findall("\")