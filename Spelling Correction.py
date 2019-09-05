
# coding: utf-8

# In[3]:


import re
import os
import math
import numpy as np
from nltk.stem import PorterStemmer
from itertools import chain
from collections import Counter
from collections import OrderedDict


# In[5]:


def wordFrequency():
    
    # *************** Implementing Spelling Correction ***************
    file_path = "C:/Users/gupta/Desktop/FIT5166 Information Retrieval Systems/example/example/doc"
    tokens = []
    # Loop for opening different files
    for xfile in os.listdir(file_path): 
        lines = []
        yfile = os.path.join(file_path, xfile)
        if os.path.isfile(yfile) and yfile.endswith('.txt'): 
            open_file = open(yfile,"r", encoding="utf-8-sig")
            lines = open_file.read()
            
            # Finding all the words
            file_tokens = re.findall(r"[A-Za-z]+",lines)
            
        tokens.append(file_tokens)
        
    # Counting occurrence of each word in a corpus
    tokens = list(chain.from_iterable(tokens))
    WORDS = Counter(tokens)
    
    return(WORDS)

WORDS = wordFrequency()

# Calculating Probability of words in a query 
def Probability(word, N=sum(WORDS.values())): 
    return WORDS[word] / N

# Correcting Words with most probable word
def correction(word): 
    word_list = []
    word_list = re.split(r" ",word)
    newWords = []
    for w in word_list:
        newWords.append(max(candidates(w), key=Probability))
    newWords = ' '.join(newWords)
    return "DO YOU MEAN?", newWords

# Generating possible candidate for a word
def candidates(word): 
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

# Generating subset of a word that appear in WORDS
def known(words): 
    return set(w for w in words if w in WORDS)

# Generating edits one edit away from a word
def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

# Generating edits two edits away from a word
def edits2(word): 
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
        


# In[8]:


query = input("Enter your query: ")
correction(query)


# In[ ]:


# *************** References ***************
# http://norvig.com/spell-correct.html

