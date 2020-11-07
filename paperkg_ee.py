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



def load_data(paper_json_path):
    print(paper_json_path)
    f1 = open(paper_json_path, "r")
    paper_dict= ujson.load(f1)
    f1.close()
    return paper_dict


'''
get stopwords
'''
def get_stop_words(lang):
    stop_words = set(stopwords.words(lang))
    return stop_words

'''
connection to the elasticsearch cluster
'''
def get_es_conn(hosts):
    es = elasticsearch.Elasticsearch(hosts)
    return es

'''
query from es with the document is wiki_entity
'''
def query_abs_over_wiki(abstract):
    return {
        "track_total_hits": "true",
        "version": "true",
        "size": 1000,
        "sort": [{
            "_score": {
                "order": "desc"
            }
        }],
        "_source": {
            
            "include": [
                "entity",
                "stem",
                "description"
            ]
        },
        "stored_fields": [
            "*"
        ],
        "script_fields": {},
        "docvalue_fields": [],
        "query": {
            "match": {
                "entity": abstract
            }
        },
        "highlight": {
            "pre_tags": [
                "@@"
            ],
            "post_tags": [
                "@@"
            ],
            "fields": {
                "*": {}
            },
            "fragment_size": 2147483647
        }
    }

'''
detect useless entity
'''
def useless_detect(entity):
    fh = "~!@#$%^&*()_+-*/<>,.[]\/ "
    sz = "0123456789"
    flag = 0
    for e in entity:
        if e in fh or e in sz:
            flag += 1
    if flag == len(entity):
        return True
    else:
        return False

'''
detect the word after stemming wether appearing continious
'''
def check_entity_in_qa(a,b):
    for i in range(len(b)-len(a)+1):
        flag = True
        if b[i]==a[0]:
            for j in range(len(a)-1):
                if b[i+j+1] != a[j+1]:
                    flag = False
            if  not flag:
                continue
            return i+1
    return False

'''
check if the entity is the full highlight and meaningful entity
'''
def checkfully(entity, st_qa, stop_words):# entity=>hit["_source"]["stem"]
    st_entity = entity.split(' ')
    pos = check_entity_in_qa(st_entity,st_qa) # pos is the first list index of the matched word +1
    if not pos:
        return False
    elif entity not in stop_words:
        if useless_detect(entity):
            return False
        else:
            return pos
    else:
        return False
    
'''
get entity position
'''
def get_entity_position(source, qa, st_qa, pid_abs, pos): #entity-->hit["_source"]
    qaInAbs_pos = pid_abs.find(qa)
    enInQa_pos = 0
    qa = qa.split(' ')
    for i in range(len(qa)):
        if i<pos-1:
            enInQa_pos += len(qa[i])
    entity_len = 0
    entity_name_in_qa = qa[pos-1:pos-1+len(source['entity'].split(' '))]
    for j in range(len(entity_name_in_qa)):
        entity_len += len(entity_name_in_qa[j])
    return qaInAbs_pos+enInQa_pos+pos-1, qaInAbs_pos+enInQa_pos+pos-1+entity_len+len(entity_name_in_qa)-1


def e_distance(v1,v2):
    return np.sqrt(np.sum(np.square(v1-v2)))

def sim(p,Qid,index,corpus_query):
    p_index = corpus_query[str(p)]
    Q_index = corpus_query[str(Qid)]
    sim_score = (index.vector_by_id(p_index)).multiply(index.vector_by_id(Q_index)).sum()   #点乘计算相似度
    return float(sim_score)

def get_score_paper_entityDescription_and_entityLength(pid,mr_key,entity_Qlist,index,corpus_query):
    abs_Q_score = []
    title_Q_score = []
    mr_Q_score = []
    en_word_len = []
    en_letter_len = []
    complex_len = []
    for entity in entity_Qlist:
        Q = entity[0]
        abs_Q_score.append(sim((pid,'abstract'),Q,index,corpus_query))
        title_Q_score.append(sim((pid,'title'),Q,index,corpus_query))
        mr_Q_score.append(sim((pid,mr_key),Q,index,corpus_query))
        en_letter_len.append(len(entity[1]))
        en_word_len.append(len(entity[1].split(' ')))
        complex_num = 0
        for e in entity[1]:
            if e not in string.ascii_letters:
                complex_num += 1
        complex_len.append(complex_num)
    return abs_Q_score,title_Q_score,mr_Q_score,en_word_len,en_letter_len,complex_len

def print_paperkg_top3(paperkg_v2, paper_dict):
    paperkg_final = {}
    p = list(paperkg_v2.keys())[0]
    paperkg_final[p]={}
    paperkg_final[p]['abstract'] = paper_dict[p]['abstract']
    paperkg_final[p]['title'] = paper_dict[p]['title']
    if "author" not in list(paper_dict[p].keys()):
        paperkg_final[p]['author'] = ""
    else:
        paperkg_final[p]['author'] = paper_dict[p]['author']
    for q in paperkg_v2[p].keys():
        paperkg_final[p][q]=[]
        for index in range(0, len(paperkg_v2[p][q])):
            if index >= 3:
                break
            paperkg_final[p][q].append(paperkg_v2[p][q][index])

    return paperkg_final

def write_paperkg_csv(paperkg_v2, paper_dict, csv_path):
    columns = ["paper_id","question","origin_res","res_entity","entity_name","entity_startPos","entity_endPos","entity_description","score","abs_score","title_score","qa_score","word_len","letter_len","complex_len"]
    #f_csv = open(csv_path,'w',encoding='utf8',newline='')
    #csv_writer = csv.writer(f_csv)
    #csv_writer.writerow(["paper_id","question","origin_res","res_entity","entity_name","entity_description","entity_startPos","entity_endPos","score","abs_score","title_score","qa_score","word_len","letter_len","complex_len"])
    data = []
    pid = list(paperkg_v2.keys())[0]
    for q in paperkg_v2[pid].keys():
        for index in range(0, len(paperkg_v2[pid][q])):
            if index >= 3:
                break
            try:
                row = [pid,
                       q,
                       paper_dict[pid][q],
                       paperkg_v2[pid][q][index][0][0],
                       paperkg_v2[pid][q][index][0][1],
                       paperkg_v2[pid][q][index][0][2],
                       paperkg_v2[pid][q][index][0][3],
                       paperkg_v2[pid][q][index][0][4],
                       paperkg_v2[pid][q][index][1],
                       paperkg_v2[pid][q][index][2],
                       paperkg_v2[pid][q][index][3],
                       paperkg_v2[pid][q][index][4],
                       paperkg_v2[pid][q][index][5],
                       paperkg_v2[pid][q][index][6],
                       paperkg_v2[pid][q][index][7]]
                #csv_writer.writerow(row)
                data.append(row)
            except:
                print('row data is error!')
                break
    df = pd.DataFrame(data =data, columns=columns)
    #df.to_csv(csv_path,index=0)
    df.to_excel(csv_path,index=0)
    #f_csv.close()

    '''
query the paperkg_full
'''
def paperkg_init(es, paper_dict, pid, stop_words, mode="standary"):
    paperkg = {}
    mr = paper_dict[pid]
    paperkg[pid] = {}
    filter_list = []
    if mode == "parallel":
        pass
    else:
        for q in list(mr.keys())[:-3]: #[:-2]
            if mr[q] in filter_list:
                continue
            filter_list.append(mr[q])
            tmp_res = es.search(index='dbacestem', body=query_abs_over_wiki(mr[q]), request_timeout=600)["hits"]["hits"]
            print("-", q, len(tmp_res))
            st = LancasterStemmer()
            st_qa = []
            for word in mr[q].split(' '):
                st_qa.append(st.stem(word.replace('.','').replace(',','').replace('?','').replace('!','')))
            entity_filter = []
            for hit in tqdm(tmp_res):
                if hit["_id"] in entity_filter:
                    continue
                pos = checkfully(hit["_source"]["stem"], st_qa, stop_words)
                if pos:
                    pid_abs =paper_dict[pid]['abstract']
                    start_pos, end_pos =get_entity_position(hit["_source"], mr[q], st_qa, pid_abs, pos)
                    if q not in paperkg[pid].keys():
                        paperkg[pid][q] = []
                        paperkg[pid][q].append((hit["_id"], hit["_source"]["entity"], start_pos, end_pos, hit["_source"]["description"]))
                        entity_filter.append(hit["_id"])
                    else:
                        paperkg[pid][q].append((hit["_id"], hit["_source"]["entity"], start_pos, end_pos, hit["_source"]["description"]))
                        entity_filter.append(hit["_id"])
    return paperkg

'''
order the paperkg entity
'''
def paperkg_manu(paperkg_v1,index,corpus_query,stop_words,abs_w, title_w, mr_w, word_len_w, letter_len_w, complex_len_w):
    paperkg_score = {}
    pid = list(paperkg_v1.keys())[0]
    paperkg_score[pid] = {}
    mr_dict = paperkg_v1[pid]
    for mr_key,entity_Qlist in mr_dict.items():
        print(mr_key)
        abs_Q_score, title_Q_score, mr_Q_score, en_word_len,en_letter_len,complex_len = get_score_paper_entityDescription_and_entityLength(pid,mr_key,entity_Qlist,index,corpus_query) 
        final_score = list(map(lambda x:x[0]*abs_w+x[1]*title_w+x[2]*mr_w+x[3]*word_len_w+x[4]*letter_len_w+x[5]*complex_len_w,zip(abs_Q_score,title_Q_score,mr_Q_score,en_word_len,en_letter_len,complex_len)))
        final_score_entity = []
        for i,entity in enumerate(entity_Qlist):
            entity_name = entity[1]
            if abs_Q_score[i]==0 or mr_Q_score[i]<0.1:
                final_score[i] = 0
            final_score_entity.append((entity,final_score[i],abs_Q_score[i],title_Q_score[i],mr_Q_score[i],en_word_len[i],en_letter_len[i],complex_len[i]))
        paperkg_score[pid][mr_key] = sorted(final_score_entity,key=lambda s: s[2],reverse = True) # 按分数排序
    return paperkg_score
            

'''
output the final paperkg
'''
def get_paperkg_final(paper_dict, index, corpus_query, pid, field, abs_w, title_w, mr_w, word_len_w, letter_len_w, complex_len_w,x=0):
	eshosts = ['10.10.10.10:9201']
	paperkg_final = {}
	if "('{}', '{}')".format(pid,'title') in list(corpus_query.keys()):
		print("hello")
		csv_path = "./csv/%s_paper_%s_%s.xlsx"%(field,pid,x)
		start_time = time.perf_counter()
		paperkg_v1 = paperkg_init(get_es_conn(eshosts), paper_dict, pid, get_stop_words("english"))
		end_time = time.perf_counter()
		print('constracting paperkg costs '+str(end_time-start_time)+'s!')
		start_time = time.perf_counter()
		paperkg_v2 = paperkg_manu(paperkg_v1,index,corpus_query,get_stop_words("english"),abs_w, title_w, mr_w, word_len_w, letter_len_w, complex_len_w)
		end_time = time.perf_counter()
		print('caculating score costs '+str(end_time-start_time)+'s!')
		#print(csv_path)
		#write_paperkg_csv(paperkg_v2, paper_dict, csv_path)
		paperkg_final = print_paperkg_top3(paperkg_v2, paper_dict)
	else:
		print("('{}', '{}')".format(pid,'title')+' not in corpus!')
	return paperkg_final

def get_preload(paper_json_path, index_path, corpus_query_path, field):
    print('loading data!')
    start_time = time.perf_counter()
    paper_dict= load_data(paper_json_path)
    end_time = time.perf_counter()
    print('loaded paper_dict costs '+str(end_time-start_time)+'s!')
    start_time = time.perf_counter()
    index = similarities.Similarity.load(index_path)
    with open(corpus_query_path,'r', encoding='utf8') as fin:
        corpus_query  = ujson.load(fin)
    end_time = time.perf_counter()
    print('loaded index,corpus_query costs '+str(end_time-start_time)+'s!')
    return paper_dict, index, corpus_query

	