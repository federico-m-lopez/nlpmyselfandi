#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:05:47 2022

@author: ubuntu
"""

import arxiv
# import pdftotext

url = 'https://arxiv.org/pdf/2204.05979.pdf'
arxiv_id = url.split('/')[-1].replace('.pdf','')

search = arxiv.Search(id_list=["2204.05979"])
paper = next(search.results())
# filename = paper.download_pdf('./papers')

# with open(rf'./papers/{filename}', 'rb') as file:
#     pdf = pdftotext.PDF(file)

import scipdf

import re
import numpy as np
import nltk
from textblob import TextBlob

def preprocess(article):
    def generate_phrases(article):
        result = []        
        for section in article['sections']:
            txt = section['text']
            txt_vec = txt.split('\n')
            avgs = [np.average([len(i) for i in a.split(' ')]) for a in txt_vec]
            
            #clean phrases that look like formulas
            for t, avg in zip(txt_vec, avgs):
                if avg < 2.5:
                    txt = txt.replace(t[1:-1], '').replace('\n',' ')
                       
            #clean weird chars, remanents from formulas
            bad_chars = np.unique(re.sub(r'[A-Za-z0-9_.]', '', str(txt).lower().strip()).split(' '))
            # print(bad_chars)
            for bad_char in bad_chars:
                if bad_char:
                    txt = txt.replace(bad_char, '')
                    
            result.append(txt)
        return ' '.join(result)
                

    def stem_and_lem(txt):
        return txt
        lst_stopwords = nltk.corpus.stopwords.words("english")
        txt_vec = txt.split(' ')
        
        #stop words
        txt = [word for word in txt_vec if word not in lst_stopwords]
        
        #stemming
        ps = nltk.stem.porter.PorterStemmer()
        txt_vec = [ps.stem(word) for word in txt_vec]
        
        #lemmatisation
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        txt_vec = [lem.lemmatize(word) for word in txt_vec]
        
        result = ' '.join(txt_vec)
        return result
    
    return stem_and_lem(generate_phrases(article))

article_dict = scipdf.parse_pdf_to_dict('./papers/04.pdf') # return dictionary    
clean_text = preprocess(article_dict)    
    
word_count = len(clean_text.split(' '))
char_count = sum(len(word) for word in clean_text.split(' '))
sentence_count = len(clean_text.split('.'))
avg_word_length = char_count / word_count
avg_sentence_length = word_count / sentence_count
sentiment = TextBlob(clean_text).sentiment.polarity

import spacy
nlp = spacy.load("en_core_web_lg")
doc = nlp(clean_text)

with open('./test.html', 'wb') as f:
    f.write(spacy.displacy.render(doc, style="ent").encode())


import wordcloud
wc = wordcloud.WordCloud(background_color='black', max_words=100, max_font_size=35)
wc = wc.generate(str(clean_text))
cool_words = [k for k, v in wc.words_.items() if v>0.2 and len(k.replace(' ','')) > 3]


import gensim
import pandas as pd

## pre-process corpus
lst_corpus = []
for string in clean_text.split('.'):
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i + 2]) for i in range(0, 
                     len(lst_words), 2)]
    lst_corpus.append(lst_grams)
## map words to an id
id2word = gensim.corpora.Dictionary(lst_corpus)
## create dictionary word:freq
dic_corpus = [id2word.doc2bow(word) for word in lst_corpus] 
## train LDA
lda_model = gensim.models.ldamodel.LdaModel(corpus=dic_corpus, id2word=id2word, num_topics=3, random_state=123, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
   
## output
lst_dics = []
for i in range(0,3):
    lst_tuples = lda_model.get_topic_terms(i)
    for tupla in lst_tuples:
        lst_dics.append({"topic":i, "id":tupla[0], 
                         "word":id2word[tupla[0]], 
                         "weight":tupla[1]})
dtf_topics = pd.DataFrame(lst_dics, columns=['topic','id','word','weight'])


import textrank
key_phrases = textrank.extract_key_phrases(clean_text)
key_sentences = textrank.extract_sentences(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import collections

km_model = KMeans(n_clusters=5)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_text.split('.'))
feature_names = vectorizer.get_feature_names()
km_model.fit(X)


clustering = collections.defaultdict(list)
 
for idx, label in enumerate(km_model.labels_):
    clustering[label].append(idx)
 








