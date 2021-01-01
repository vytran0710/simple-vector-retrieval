import nltk
import re
import os
import pickle
from nltk.corpus import stopwords
import math
from nltk.stem import WordNetLemmatizer
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
    return filtered_words

def get_text_from_file(filename):
    with open(filename, encoding='cp1252', mode='r') as f:
        text = f.read()
    return text

def indexing(docs_path):
    terms = []
    index = []
    norm_list = []
    N = 0

    # Indexing postings
    for doc_file in sorted(os.listdir(docs_path),key=lambda x: int(os.path.splitext(x)[0])):
        filename = os.path.join(docs_path, doc_file)
        N += 1
        text = get_text_from_file(filename)
        words = get_words_from_text(text)
        for word in words:
            if word not in terms:
                terms.append(word)
                index.append([(N, 1)])
            else:
                temp_index = terms.index(word)
                if N not in [i[0] for i in index[temp_index]]:
                    index[temp_index].append((N, 1))
                    continue
                for i in range(len(index[temp_index])):
                    if index[temp_index][i][0] == N:
                        index[temp_index][i] = (N, index[temp_index][i][1] + 1)
                        break

    # Calculate weights
    for i in range(len(index)):
        for j in range(len(index[i])):
            temp = list(index[i][j])
            temp[1] = index[i][j][1] * math.log(N/len(index[i])) # Formula: TF * IDF (IDF = log(N / ndoc(t)))
            index[i][j] = tuple(temp)

    # Normalization
    for i in range(N):
        norm = 0
        for j in range(len(index)):
            temp = [item for item in index[j] if item[0] == i+1]
            if len(temp) != 0:
                norm += math.pow(temp[0][1], 2)
        norm_list.append(math.sqrt(norm))
    for i in range(len(index)):
        for j in range(len(index[i])):
            temp = list(index[i][j])
            temp[1] = index[i][j][1] / norm_list[index[i][j][0]-1]
            index[i][j] = tuple(temp)

    return terms, index

terms, index = indexing(r"D:\Github\simple-vector-retrieval\Cranfield\Cranfield")
file = open(r'D:\Github\simple-vector-retrieval\index\terms.sav', 'wb')
pickle.dump(terms, file)
file.close()
file2 = open(r'D:\Github\simple-vector-retrieval\index\index.sav', 'wb')
pickle.dump(index, file2)
file2.close()