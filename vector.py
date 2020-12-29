import nltk
import re
import os
import pickle
from nltk.corpus import stopwords
import math

def preprocess_text(text):
    processed_text = text.lower()
    processed_text = processed_text.replace("’", "'")
    processed_text = processed_text.replace("“", '"')
    processed_text = processed_text.replace("”", '"')
    non_words = re.compile(r"[^A-Za-z0-9']+")
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

def build_model(docs_path):
    terms = []
    index = []
    norm_list = []
    N = 0
    for doc_file in os.listdir(docs_path):
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
    for i in range(N):
        norm = 0
        for j in range(len(index)):
            temp = [item for item in index[j] if item[0] == i+1]
            if len(temp) != 0:
                norm += temp[0][1]*temp[0][1]
        norm_list.append(math.sqrt(norm))
    return terms, calculate_weight(index, N, norm_list)

def calculate_weight(index, N, norm_list):
    for i in range(len(index)):
            for j in range(len(index[i])):
                temp = list(index[i][j])
                temp[1] = round(index[i][j][1] * (math.log(N/len(index[i]) + 1)) / norm_list[temp[0]-1], 4)
                index[i][j] = tuple(temp)
    return index

terms, index = build_model("D:\\Courses\\CS336\\Cranfield\\Cranfield")
file = open('D:\\Courses\\CS336\\model\\terms.sav', 'wb')
pickle.dump(terms, file)
file.close()
file2 = open('D:\\Courses\\CS336\\model\\index.sav', 'wb')
pickle.dump(index, file2)
file2.close()

print(terms)
print(index)

print(len(terms))
print(len(index))