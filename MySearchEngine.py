
# coding: utf-8

# In[203]:


import re
import os
import math
import numpy as np
from nltk.stem import PorterStemmer
from itertools import chain
from collections import Counter
from collections import OrderedDict


# In[204]:


def tokenization(lines):
    # *************** Tokens by Rules ***************
    # Finding all the special tokens according to the rules provided in the specification
    special_tokens = re.findall(r"\w+-\n\w+|[\w\.-]+@[\w\.-]+|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|\B['`].+?['`]\B|[A-Z][a-z']+(?=\s[A-Z])(?:\s[A-Z][a-z']+)+|[a-zA-Z]\.[a-zA-Z]\.[a-zA-Z]|(?:[a-zA-Z]\.){2,}|\b[A-Z]+(?:\s+[A-Z]+)*\b",lines)
    # Cleaning the special tokens
    special_tokens = [w.replace('-\n', ' ') for w in special_tokens]
    special_tokens[:] = [x.strip("['+=]") for x in special_tokens]
    
    # *************** File Tokens ***************
    # Substituting implemented rules above with "" in the main file
    file_tokens = re.sub(r"\w+-\n\w+|[\w\.-]+@[\w\.-]+|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|\B['`].+?['`]\B|[A-Z][a-z']+(?=\s[A-Z])(?:\s[A-Z][a-z']+)+|[a-zA-Z]\.[a-zA-Z]\.[a-zA-Z]|(?:[a-zA-Z]\.){2,}|\b[A-Z]+(?:\s+[A-Z]+)*\b","",lines)
    # Cleaning the file tokens
    file_tokens = re.split(r"( )|\n|\-\-|\[|\]|\.|\,|\:|\;|\"|\'|\(|\)|\?|\!|\}|\{|\/",file_tokens)
    file_tokens = [x for x in file_tokens if x != None and x!=' ' and x!='']
    file_tokens[:] = [x.strip('[-+=]') for x in file_tokens]

    return special_tokens, file_tokens


# In[205]:


def stopwordsRemoval(file_tokens):
    # *************** Removing Stop Words ***************
    stop_file = open(stopwords,"r")
    stop = stop_file.read()
    stop = stop.split('\n')
    file_tokens = [w for w in file_tokens if w not in stop]

    return(file_tokens)


# In[206]:


def stemming(file_tokens):
    # *************** Stemming of Words ***************
    psfile = []
    ps = PorterStemmer()
    for word in file_tokens:
        psfile.append(ps.stem(word))

    return psfile


# In[207]:


def indexing():
    # Initialising lists and dictionaries
    docs = []
    termsCollection = []
    term_freq_dic = {}
    Inverse_document_freq = {}
    
    # Loop for opening different files
    for xfile in os.listdir(file_path): 
        lines = []
        yfile = os.path.join(file_path, xfile)
        if os.path.isfile(yfile) and yfile.endswith('.txt'): 
            open_file = open(yfile,"r", encoding="utf-8-sig")
            lines = open_file.read()
            
            # Calling Tokenization function
            special_tokens, file_tokens = tokenization(lines)
    
            # Converting tokens to Lower Case
            special_tokens[:] = [x.lower() for x in special_tokens]
            file_tokens[:] = [x.lower() for x in file_tokens]
            
            # Calling Stop Words Removal function
            file_tokens = stopwordsRemoval(file_tokens)
            
            # Calling Stemming function
            file_tokens = stemming(file_tokens)
            
            # Joining both token files
            tokens = special_tokens + file_tokens
            
            # *************** Term Frequency Calculation ***************
            term_freq = dict(Counter(tokens))
            term_freq_dic[xfile] = term_freq
        
            # Appending Document keys for Document Frequencies
            docs.append(list(set(term_freq.keys())))
            
            
    # *************** Inverse Document Frequency Calculation ***************
    N = len(docs)
    
    # Flattening the list of lists
    docs = list(chain.from_iterable(docs))
    
    # Document frequency
    document_freq = dict(Counter(docs))
    
    
    # Inverse Document Frequency
    Inverse_document_freq.update((k, round(np.log(N / (float(v) + 1)),3)) for k,v in document_freq.items())
    
    # Vocabulary
    vocab = Inverse_document_freq.keys()
    vocab = sorted(vocab)

    # *************** Creating Inverted Index ***************
    ind = []
    term_ind = []
    
    for idf in Inverse_document_freq:
        docs_terms = []
        docs_terms.append(idf)
        for k,v in term_freq_dic.items():
            if idf in v:
                docs_terms.append(k)
                docs_terms.append(str(v[idf]))
        docs_terms.append(str(Inverse_document_freq[idf]))
        docs_terms = ','.join(docs_terms)
        term_ind.append(docs_terms)

        
    # Writing Inverted Index to a text file
    with open("C:/Users/gupta/Desktop/FIT5166 Information Retrieval Systems/inverted.txt", 'w') as filehandle:  
        for listitem in term_ind:
            filehandle.write('%s\n' % listitem)
    
    
    # Writing Vocabulary to a text file
    with open("C:/Users/gupta/Desktop/FIT5166 Information Retrieval Systems/vocab.txt", 'w') as filehandle:  
        for listitem in vocab:
            filehandle.write('%s\n' % listitem)


    


# In[208]:


def Search():


    # *************** Query Searching ***************
    
    # Open and read Inverted and Vocabulary file
    new_inverted_file = open(inverted_file,"r", encoding="utf-8-sig")
    new_vocab_file = open(vocab_file,"r", encoding="utf-8-sig")
    inverted = new_inverted_file.read()
    vocab = new_vocab_file.read()
    # Convert both of them into different lists
    inverted = re.split(r"\n",inverted)
    vocab = re.split(r"\n",vocab)
    inverted = [re.split(r",",i) for i in inverted]
    
    
    # *************** Creating Document Vector ***************
    
    # Finding weights of terms in all the documents
    docs = []
    for i in inverted:
        for x in range(1,len(i)-1):
            if x%2 != 0:
                docs.append(i[x])
            else:
                i[x] = float(i[x])
                i[x] = round(float(i[len(i)-1]) * i[x],3)
     
    docs = sorted(list(set(docs)))
    inverted = sorted(inverted)
    
    # Converting inverted index to a dictionary
    doc_vector = {}
    for i in inverted:
        doc_vector_dic = {}
        for x in range(1,len(i)-1,2):
            doc_vector_dic[i[x]] = i[x+1]
        doc_vector[i[0]] = doc_vector_dic
    
    # Document Vector
    doc_vec = dict()
    for d in docs:
        weights = []
        for term in vocab:
            if d in doc_vector[term]:
                weights.append(doc_vector[term][d])
            else:
                weights.append(0)
        doc_vec[d] = weights
    
    
    # *************** Creating Query Vector ***************
    query_tf = []
    # Taking input from the user for query and number of documents to be listed
    q = input("Enter your query: ")
    num_docs = int(input("Enter number of documents: "))
    
    # Calling tokenization on query
    special_query_tokens, query_tokens = tokenization(q)
    
    # Converting tokens to Lower Case
    special_query_tokens[:] = [x.lower() for x in special_query_tokens]
    query_tokens[:] = [x.lower() for x in query_tokens]

    # Calling Stop Words Removal function
    query_tokens = stopwordsRemoval(query_tokens)

    # Calling Stemming function
    query_tokens = stemming(query_tokens)

    # Joining both files
    tokens = special_query_tokens + query_tokens

    # Calculating Term Frequencies for the query
    query_term_freq = dict(Counter(tokens))

    # Dictionary containing weights
    qw = {}
    for k,v in query_term_freq.items():
        for i in inverted:
            if k in i:
                qw[k] = round((v * float(i[-1])),3)
    
    # Appending 0 for non existing terms in a query
    for v in vocab:
        if v not in qw.keys():
            qw[v] = 0
            
    qw = sorted(qw.items())
    
    # Query Vector
    qv = []
    for q in qw:
        qv.append(q[1])
    # *************** Calculating Cosine Similarity ***************
    
    # Calculating Numerator 
    p = {}
    for k,d in doc_vec.items():
        n = [round(a*b,3) for a,b in zip(d,qv)]
        p[k] = n

    numerator = {}
    for k,e in p.items():
        numerator[k] = round(sum(e),3)
    
    
    # Calculating Denominator
    
    # sum of squares of query vector
    s_sq = []
    for q in qv:
        s_sq.append(round(q**2,3))
    sum_sq = round(math.sqrt(sum(s_sq)),3)
    
    # sum of squares of document vector
    d_sq = []
    sum_d_sq = {}
    for k,dd in doc_vec.items():
        for d in dd:
            d_sq.append(round(d**2,3))
        sum_d_sq[k] = (round(math.sqrt(sum(d_sq)),3))
    
    # denominator
    denominator = {}
    for k,i in sum_d_sq.items():
        denominator[k] = round(i * sum_sq,3) 

    # Cosine Similarity
    cosine = {}
    for k,d in denominator.items():
        cosine[k] = round(numerator[k]/denominator[k],3)
        
    cosine_similarity = sorted(cosine.items(),reverse=True, key=lambda x: x[1])[:num_docs]

    return (cosine_similarity)


# In[209]:


def main():
    index = indexing()
    search = Search()
    print(search)    


# In[210]:


if __name__ == "__main__":
    file_path = "C:/Users/gupta/Desktop/FIT5166 Information Retrieval Systems/example/example/doc"
    stopwords = "C:/Users/gupta/Desktop/FIT5166 Information Retrieval Systems/StopwordsNew.txt"
    inverted_file = "C:/Users/gupta/Desktop/FIT5166 Information Retrieval Systems/inverted.txt"
    vocab_file = "C:/Users/gupta/Desktop/FIT5166 Information Retrieval Systems/vocab.txt"
    main()

