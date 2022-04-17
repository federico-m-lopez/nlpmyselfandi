# -*- coding: utf-8 -*-
import spacy
import nltk

nltk.download('stopwords')
spacy.cli.download("en_core_web_lg")
spacy.cli.download("en_core_web_sm")

import scipdf
import re
import numpy as np
import gensim
import pandas as pd
import summa

from textblob import TextBlob
from sorcery import dict_of
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import collections
import matplotlib.pyplot as plt

class DataModel:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.path = kwargs.get('path')
        self.file = kwargs.get('file')
        self.nlp = spacy.load("en_core_web_lg")
        
        #warmup
        self.init()
        
    def init(self):
        self.filename = self.path + '/' + self.file
        #todo: enable before production
        #self.article = scipdf.parse_pdf_to_dict(self.filename, grobid_url='https://cloud.science-miner.com/grobid/') 
        self.article = scipdf.parse_pdf_to_dict(self.filename) 
        self.clean_text = self.preprocess()
        self.doc = self.nlp(self.clean_text)
        self.data = self.generate_data()
        
    
    def preprocess(self):
        #todo: transform to pipleine
        temp_txt = self.__preproces_article_structure()
        temp_txt = self.__preprocess_stem_and_lem(temp_txt)
        return temp_txt
            
    
    def __preproces_article_structure(self):
        result = []        
        for section in self.article['sections']:
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
    
    def __preprocess_stem_and_lem(self, txt):
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
    
    def generate_data(self):
        result = {}
        result['stat'] = self.generate_stat_data()
        result['wc'] = self.generate_wordcloud_data()
        result['lda'] = self.generate_lda_data()
        result['tr'] = self.generate_summarized_data()
        result['kmeans'] = self.generate_kmeans_data()
        
        return result

    def generate_stat_data(self):
        word_count = int(len(self.clean_text.split(' ')))
        char_count = int(sum(len(word) for word in self.clean_text.split(' ')))
        sentence_count = int(len(self.clean_text.split('.')))
        avg_word_length = char_count / word_count
        avg_sentence_length = word_count / sentence_count
        sentiment = TextBlob(self.clean_text).sentiment.polarity
        
        return dict(word_count=word_count, char_count=char_count, sentence_count=sentence_count, avg_word_length=avg_word_length, avg_sentence_length=avg_sentence_length, sentiment=sentiment)
        
        
    def render_spacy_html(self, filename='./test.html'):
        with open('./test.html', 'wb') as f:
            f.write(spacy.displacy.render(self.doc, style="ent").encode())
    
    def generate_wordcloud_data(self):
        wc = WordCloud(background_color='black', max_words=100, max_font_size=50, width=800, height=400, margin=0)
        wc = wc.generate(str(self.clean_text))
        wordcloud = [k for k, v in wc.words_.items() if v>0.2 and len(k.replace(' ','')) > 3]
        
        fig, ax = plt.subplots(figsize = (12, 8), dpi=1200)
        ax.imshow(wc, interpolation = 'bilinear')
        plt.axis('off')
        self.wc_fig = fig
        
        return dict(wordcloud=wordcloud)
    
    def generate_lda_data(self):
        ## pre-process corpus
        n_ngrams = 3
        n_topics = 5
        
        lst_corpus = []
        for string in self.clean_text.split('.'):
            lst_words = string.split()
            lst_grams = [" ".join(lst_words[i:i + n_ngrams]) for i in range(0,len(lst_words), n_ngrams)]
            lst_corpus.append(lst_grams)
        ## map words to an id
        id2word = gensim.corpora.Dictionary(lst_corpus)
        ## create dictionary word:freq
        dic_corpus = [id2word.doc2bow(word) for word in lst_corpus] 
        ## train LDA
        lda_model = gensim.models.ldamodel.LdaModel(corpus=dic_corpus, id2word=id2word, 
                                                    num_topics=n_topics, random_state=123, 
                                                    update_every=1, chunksize=100, passes=10, 
                                                    alpha='auto', per_word_topics=True)
           
        ## output
        lst_dics = []
        for i in range(0,n_topics):
            lst_tuples = lda_model.get_topic_terms(i)
            for tupla in lst_tuples:
                lst_dics.append({"topic":i, "id":tupla[0], 
                                 "word":id2word[tupla[0]], 
                                 "weight":tupla[1]})
        self.dtf_topics = pd.DataFrame(lst_dics, columns=['topic','id','word','weight'])
        return self.dtf_topics.to_dict()

    def generate_summarized_data(self):
        keywords = summa.keywords.keywords(self.clean_text, words=20, split=True)
        summary = summa.summarizer.summarize(self.clean_text, words=150, split=True)
        return dict(keywords=keywords, summary=summary)


    def generate_kmeans_data(self):
        
        km_model = KMeans(n_clusters=5)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.clean_text.split('.'))
        feature_names = vectorizer.get_feature_names()
        km_model.fit(X)
        clustering = collections.defaultdict(list)
        centers = km_model.cluster_centers_
        
        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)
            
        return dict(feature_names=feature_names, clustering=clustering, centers=centers)
                       