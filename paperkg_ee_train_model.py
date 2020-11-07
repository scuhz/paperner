import math
import csv
import os
import pandas as pd
import elasticsearch
import elasticsearch.helpers
import ujson
#from stop_words import get_stop_words
from tqdm import tqdm
import time
from nltk.corpus import stopwords
import numpy as np
import string
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora
from gensim import models
from gensim import similarities



'''
get stopwords
'''
def get_stop_words(lang):
    stop_words = set(stopwords.words(lang))
    return stop_words


'''
load dictionary(one time action)
'''
def load_data(paper_json_path, wiki_json_path):
    print(paper_json_path, wiki_json_path)
    f1 = open(paper_json_path, "r")
    paper_dict= ujson.load(f1)
    f1.close()

    f2 = open(wiki_json_path, "r")
    wiki_tiny_dict = ujson.load(f2)
    f2.close()

    return paper_dict, wiki_tiny_dict

'''
state paper corpus
'''
def get_corpus_paper_entityDescription(paper_dict,wiki_tiny_dict, stop_words):
    corpus = []
    corpus_query = {}
    index = 0
    for pid in tqdm(paper_dict.keys()):# 添加所有paper的title，abs，mr入corpus
        p_mr_info = paper_dict[pid]
        for mr_key,mr_key_content in p_mr_info.items():
            if mr_key == "author":
                continue
            p_mr_key_content_after_stop_stemmer = []
            st = LancasterStemmer()
            for word in mr_key_content.split(' '):
                if word not in stop_words:
                    p_mr_key_content_after_stop_stemmer.append(st.stem(word))
            corpus.append(p_mr_key_content_after_stop_stemmer)
            corpus_query[(pid,mr_key)] = index
            index += 1
    for Qid in tqdm(wiki_tiny_dict.keys()):
        Qdescription = wiki_tiny_dict[Qid]["description"]
        Qdescription_after_stop_stemmer = []
        st = LancasterStemmer()
        for word in Qdescription.split(' '):
            if word not in stop_words:
                Qdescription_after_stop_stemmer.append(st.stem(word))
        corpus.append(Qdescription_after_stop_stemmer)
        corpus_query[Qid] = index
        index += 1
    return corpus,corpus_query

'''
get tfidf weight
'''
def train_tfidf_to_file(paper_json_path, corpus_query_path, index_path, stop_words,  wiki_json_path):
    paper_dict, wiki_tiny_dict = load_data(paper_json_path, wiki_json_path)
    print('constracting corpus from json!')
    corpus,corpus_query= get_corpus_paper_entityDescription(paper_dict,wiki_tiny_dict,stop_words)
    dictionary = corpora.Dictionary(corpus) #corpus中所有词
    bow_corpus = []
    for text in tqdm(corpus):
        bow_corpus.append(dictionary.doc2bow(text)) #corpus变换成数值型的词袋模型
    # train the model
    tfidf = models.TfidfModel(bow_corpus)
    print('corpus constraction finished!')
    with open(corpus_query_path, 'w', encoding='utf8')as f_out:
        ujson.dump(corpus_query, f_out)
    f_out.close()
    print('start to constract index of corpus!')
    index = similarities.Similarity(index_path,tfidf[bow_corpus], num_features=len(dictionary))
    index.save()
    print('index constraction finished!')
    return index,corpus_query




if __name__=='__main__':
    raw_path = "."
    paper_json_path= os.path.join(raw_path,"data/dde_paper_v4.json")   ##title, abs , author
    #wiki_json_path = os.path.join(raw_path,"data/wiki_slim.json")
    wiki_json_path = "/home/daven/acekg/dbpedia/dbace_dict_v1.json"     ###entity's description
    index_path = os.path.join(raw_path,"output/dde/db_dde_sim")
    corpus_query_path = os.path.join(raw_path,"output/db_dde_corpus_query.json")
    stop_words = get_stop_words('english')
    train_tfidf_to_file(paper_json_path, corpus_query_path, index_path, stop_words,wiki_json_path)