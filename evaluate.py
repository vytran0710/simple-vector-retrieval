import nltk
import re
import os
import pickle
from nltk.corpus import stopwords
import math
from nltk.stem import WordNetLemmatizer
from collections import Counter
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = processed_text.replace("’", "'")
    processed_text = processed_text.replace("“", '"')
    processed_text = processed_text.replace("”", '"')
    non_words = re.compile(r"[^A-Za-z']+")
    processed_text = re.sub(non_words, ' ', processed_text)
    return processed_text

def get_words_from_text(text):
    processed_text = preprocess_text(text)
    filtered_words = [word for word in processed_text.split() if word not in stopwords.words('english')]
    processed_words = []
    for i in range(len(filtered_words)):
        processed_words.append(lemmatizer.lemmatize(filtered_words[i]))
    return processed_words

def get_search_results(query, terms, index):
    words = get_words_from_text(query)

    # Get postings
    temp_postings = {}
    temp_words = list(set(words))
    for i in range(len(temp_words)):
        if temp_words[i] in terms:
            temp_postings[temp_words[i]] = index[terms.index(temp_words[i])].copy()

    # Calculating TF * IDF (IDF = 1 / ndoc(t))
    temp = [item for item in words if item in terms]
    weight_list = Counter(temp)
    weight_list = {x: weight_list[x] * (1 / len(temp_postings[x])) for x in weight_list}
    
    # Normalization
    norm = 0
    temp = list(weight_list.values())
    for i in range(len(temp)):
        norm += math.pow(temp[i], 2)
    norm = math.sqrt(norm)
    weight_list = {x: weight_list[x] / norm for x in weight_list}

    for i in temp_postings:
        for j in range(len(temp_postings[i])):
            temp = list(temp_postings[i][j])
            temp[1] *= weight_list[i]
            temp_postings[i][j] = tuple(temp)

    # Get search result(s)
    res = list(temp_postings.values())
    result = {}
    for L in res:
        for key, value in L:
            result[key] = result.get(key, 0) + value
    result = list(result.items())

    return sorted(result, key=lambda tup: tup[1], reverse=True)

file = open(r"D:\Github\simple-vector-retrieval\index\terms.sav", 'rb')
terms = pickle.load(file)
file2 = open(r"D:\Github\simple-vector-retrieval\index\index.sav", 'rb')
index = pickle.load(file2)
print(get_search_results("what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft", terms, index))