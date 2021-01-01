import nltk
import re
import os
import pickle
from nltk.corpus import stopwords
import math
from nltk.stem import WordNetLemmatizer
from collections import Counter
import bisect
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

    # Calculating TF * IDF (IDF = log((N / ndoc(t)) + 1)))
    N = []
    for i in range(len(index)):
        N.append(max(index[i],key=lambda item:item[0])[0])
    N = max(N)
    temp = [item for item in words if item in terms]
    weight_list = Counter(temp)
    weight_list = {x: weight_list[x] * math.log((N / len(temp_postings[x])) + 1) for x in weight_list}
    
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

def evaluateMAP(test_query, test_path, terms, index):
    with open(test_query, encoding='cp1252', mode='r') as f:
        temp = f.readlines()
    temp = [x.strip().split('\t') for x in temp]
    query = []
    for i in range(len(temp)):
        query.append(temp[i][1])
    goal = []
    for doc_file in sorted(os.listdir(test_path),key=lambda x: int(os.path.splitext(x)[0])):
        filename = os.path.join(test_path, doc_file)
        with open(filename, encoding='cp1252', mode='r') as f:
            temp2 = f.readlines()
        temp2 = [x.strip().split() for x in temp2]
        temp3 = []
        for i in range(len(temp2)):
            temp3.append(int(temp2[i][1]))
        goal.append(temp3)
    average_precision = []
    for i in range(len(goal)):
        temp4 = get_search_results(query[i], terms, index)
        temp5 = [i[0] for i in temp4]
        precision = []
        recall = []
        count = 0
        n = 0
        temp_precision = []
        for j in range(len(temp5)):
            if temp5[j] in goal[i]:
                count += 1
            n += 1
            recall.append(count / len(goal[i]))
            precision.append(count / n)
        for j in range(10):
            if j/10 < max(recall):
                temp6 = bisect.bisect_right(recall, j/10)
                temp_precision.append(max(precision[temp6:]))
        if len(temp_precision) != 0:
            average_precision.append(sum(temp_precision) / 10)
        else:
            average_precision.append(0)
    if len(average_precision) != 0:
        return sum(average_precision) / len(average_precision)
    else:
        return 0
    
file = open(r"D:\Github\simple-vector-retrieval\index\terms.sav", 'rb')
terms = pickle.load(file)
file2 = open(r"D:\Github\simple-vector-retrieval\index\index.sav", 'rb')
index = pickle.load(file2)
print(evaluateMAP(r"D:\Github\simple-vector-retrieval\Cranfield\TEST\query.txt", r"D:\Github\simple-vector-retrieval\Cranfield\TEST\RES", terms, index))