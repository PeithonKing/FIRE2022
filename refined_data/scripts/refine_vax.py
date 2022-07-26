import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
pd.set_option('max_colwidth', 150)
from tqdm import tqdm
# import os
# import time
import json
tokenizer = nltk.RegexpTokenizer(r"\w+")
ps = PorterStemmer()
file = "../../raw_data/vax_train.csv"

with open("words_dictionary.json") as f: D1 = sorted(list(json.load(f).keys()))

manual = ["covid", "lockdown"]
# words = list(set([ps.stem(x.lower()) for x in set(list(words.words()) + list(english_words_set) + manual + D1)]))
words = list(set([ps.stem(x.lower()) for x in D1+manual]))

def process(string,
            tokenizer = nltk.RegexpTokenizer(r"\w+"),
            ps = PorterStemmer(),
            stopwords = stopwords.words('english')):
    '''
    - A function to process a string and return a list of tokens.
    - We tokenize the string, remove stopwords and numbers, and
        finally stem the tokens to keep them in a list.
    - This function will be used in all cases uniformly so that 
        we can compare "APPLES WITH APPLES".
    '''
    global inters
    a = string.split()
    # print(a)
    string = [
                word.lower() for word in a 
                if len(word) and
                   not word.startswith("http") and
                   word[0] != '@' and
                   word[0] != '#' and
                   word[0] != '.' and
                   word[0] != '!' and
                   word[0] != '?' and
                   word[0] != '\n' and
                   word[0] != '\t'
    ]
    string = " ".join(string)
    string = tokenizer.tokenize(string.lower()) # tokenize
    tokens = [
                # ps.stem(fl) for fl in string  # stem tokens
                fl for fl in string  # stem tokens
                if not fl.isnumeric() and  # remove numbers
                fl not in stopwords and  # remove stopwords
                fl in words
    ]
    # for x in tokens:
    #     if x.lower() not in words:
    #         inters.append(x)
    return " ".join(tokens)  # returns processed string

data = pd.read_csv(file).T.to_dict()

inters = []
for key, value in tqdm(data.items()):
    data[key]["tweet"] = process(value["tweet"])

pd.DataFrame(data).T.to_csv("data.csv", index=False)