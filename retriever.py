import pickle
from nltk.corpus import stopwords
import re
import math

file = open("D:\\Github\\simple-vector-retrieval\\temp\\terms.sav", 'rb')
terms = pickle.load(file)
file2 = open("D:\\Github\\simple-vector-retrieval\\temp\\index.sav", 'rb')
index = pickle.load(file2)

query = input()

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

words = list(set(get_words_from_text(query)))

search_list = []
for i in range(len(words)):
    if words[i] in terms:
        search_list.append(index[terms.index(words[i])].copy())


def get_search_result(search_list):
    norm = 0
    norm_list = []
    for i in range(len(search_list)):
        norm += pow(len(search_list[i]), 2)
    norm = math.sqrt(norm)
    for i in range(len(search_list)):
        norm_list.append(len(search_list[i]) / norm)
    res = []
    for i in range(len(search_list)):
        for j in range(len(search_list[i])):
            if search_list[i][j][0] not in [i[0] for i in res]:
                res.append(search_list[i][j])
                continue
            for k in range(len(res)):
                if res[k][0] == search_list[i][j][0]:
                    temp = list(search_list[i][j])
                    temp2 = list(res[k])
                    temp2[1] = round(temp[1] * norm_list[i] + temp2[1], 4)
                    res[k] = tuple(temp2)
                    break
                elif res[k][0] > search_list[i][j][0]:
                    temp = list(search_list[i][j])
                    temp[1] = round(temp[1] * norm_list[i], 4)
                    res.insert(k, tuple(temp))
                    break
    return res

print(get_search_result(search_list))
print(len(get_search_result(search_list)))