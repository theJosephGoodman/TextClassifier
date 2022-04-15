#!/usr/bin/env python
# coding: utf-8

# In[14]:


LOGREG_ROOT = './fited_logreg.sav' #путь к сохраненной модели логистической регрессии
INPUT_ROOT = './input.txt' #путь к txt файлу в котором в каждой строке хранится анализируемое предложение
BERT_ROOT = './rubert-base-cased-sentence' #путь к модели rubert


import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pymorphy2
import torch

from transformers import AutoTokenizer, AutoModel




def preprocess_df(df, stopwords, lemmatizer):
    '''
    df: датафрейм с тренировочными данными без столбца таргетов
    stopwords: список стоп-слов
    lemmatizer: лемматизатор от pymorphy2

    функция проводит предобработку сырого текста
    '''
    
    # приведение к нижнему регистру и удаление стоп-слов
    x_filtered = df.title.str.lower().str.split(' ').apply(lambda x: [i for i in x if (i.isalpha() & ~(i in stopwords))])
    x_filtered = x_filtered[x_filtered.apply(lambda x: len(x)>0)]
    
    
    # лемматизация
    x_lemmatized = x_filtered.apply(lambda x: [(lemmatizer.parse(word)[0]).normal_form for word in x ])

    return x_lemmatized

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

STOPWORDS = stopwords.words('russian')
LEMMATIZER = pymorphy2.MorphAnalyzer()
TOKENIZER = AutoTokenizer.from_pretrained(BERT_ROOT)
MODEL = AutoModel.from_pretrained(BERT_ROOT)


with open(LOGREG_ROOT, 'rb') as file:
    logreg = pickle.load(file)

with open(INPUT_ROOT, encoding='utf-8') as file:
    lines = [line.rstrip().lstrip() for line in file]
df = pd.DataFrame(lines, columns=['title'])





prep_text = preprocess_df(df, STOPWORDS, LEMMATIZER).str.join(' ')
num_words = prep_text.apply(lambda x: len(x)).values[:, np.newaxis]

embeddings = [embed_bert_cls(row, MODEL, TOKENIZER) for row in prep_text]
matrix  = np.hstack((embeddings,
                    num_words))

    
prediction = logreg.predict(matrix)

with open('output.txt', 'w') as f:
    for item in prediction:
        f.write("%s\n" % item)

