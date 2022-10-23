import regex
from pprint import pprint
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import sklearn
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize,sent_tokenize

file = open("data/corpus.txt","r")
corpus=file.read()

PATTERN = r"[\w']+|^\W"

word_tokens=regex.findall(PATTERN, corpus)
#word_tokens=word_tokenize(corpus)
word_tokens=[w.lower() for w in word_tokens]
#train, test = train_test_split(word_tokens, test_size = 0.1)

voacbulary=list(set(word_tokens))

gram1_counts={}
gram2_counts={}
gram3_counts={}
gram4_counts={}
gram5_counts={}

for i in range(len(word_tokens)-1):
  gram1=(word_tokens[i])
  if gram1 in gram1_counts.keys():
    gram1_counts[gram1]+=1
  else:
    gram1_counts[gram1]=1

for i in range(len(word_tokens)-2):
  gram2=(word_tokens[i],word_tokens[i+1])
  if gram2 in gram2_counts.keys():
    gram2_counts[gram2]+=1
  else:
    gram2_counts[gram2]=1

for i in range(len(word_tokens)-3):
  gram3=(word_tokens[i],word_tokens[i+1],word_tokens[i+2])
  if gram3 in gram3_counts.keys():
    gram3_counts[gram3]+=1
  else:
    gram3_counts[gram3]=1

for i in range(len(word_tokens)-4):
  gram4=(word_tokens[i],word_tokens[i+1],word_tokens[i+2],word_tokens[i+3])
  if gram4 in gram4_counts.keys():
    gram4_counts[gram4]+=1
  else:
    gram4_counts[gram4]=1

for i in range(len(word_tokens)-5):
  gram5=(word_tokens[i],word_tokens[i+1],word_tokens[i+2],word_tokens[i+3],word_tokens[i+4])
  if gram5 in gram5_counts.keys():
    gram5_counts[gram5]+=1
  else:
    gram5_counts[gram5]=1

def suggest_next_word(input,gram):
  vocab_probabilities={}
  tokenized_input=word_tokenize(input.lower())
  if(gram==2):
    input_gram=tokenized_input[-2:]
    for vocab_word in voacbulary:
      test_output_gram=(input_gram[0],input_gram[1],vocab_word)
      test_input_gram=(input_gram[0],input_gram[1])
      test_output_gram_count=gram3_counts.get(test_output_gram,0)
      test_input_gram_count=gram2_counts.get(test_input_gram,0)
      if(test_input_gram_count==0):
        probability=0
      else:
        probability=test_output_gram_count/test_input_gram_count
      vocab_probabilities[vocab_word]=probability
  elif(gram==3):
    input_gram=tokenized_input[-3:]
    for vocab_word in voacbulary:
      test_output_gram=(input_gram[0],input_gram[1],input_gram[2],vocab_word)
      test_input_gram=(input_gram[0],input_gram[1],input_gram[2])
      test_output_gram_count=gram4_counts.get(test_output_gram,0)
      test_input_gram_count=gram3_counts.get(test_input_gram,0)
      if(test_input_gram_count==0):
        probability=0
      else:
        probability=test_output_gram_count/test_input_gram_count
      vocab_probabilities[vocab_word]=probability
  elif(gram==4):
    input_gram=tokenized_input[-4:]
    for vocab_word in voacbulary:
      test_output_gram=(input_gram[0],input_gram[1],input_gram[2],input_gram[3],vocab_word)
      test_input_gram=(input_gram[0],input_gram[1],input_gram[2],input_gram[3])
      test_output_gram_count=gram5_counts.get(test_output_gram,0)
      test_input_gram_count=gram4_counts.get(test_input_gram,0)
      if(test_input_gram_count==0):
        probability=0
      else:
        probability=test_output_gram_count/test_input_gram_count
      vocab_probabilities[vocab_word]=probability
  
  top_suggestions=sorted(vocab_probabilities.items(),key=lambda x: x[1],reverse=True)[:10]
  return top_suggestions

print(suggest_next_word("a deep well and",4))
#print(gram2_counts[('tortoise','exclaimed')])
#print(gram2_counts)