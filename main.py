from fastapi import FastAPI
from paperkg_ee import *
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###########
### DDE ###
###########
raw_path = "."
dde_paper_json_path= os.path.join(raw_path,"data/dde_paper_v4.json")#,os.path.join(raw_path,"data/CS_dic_1017.json")
dde_index_path = os.path.join(raw_path,"output/dde/db_dde_sim") #output/dde/dde_sim"
dde_corpus_query_path = os.path.join(raw_path,"output/db_dde_corpus_query.json")
dde_paper_dict, dde_index, dde_corpus_query = get_preload(dde_paper_json_path, dde_index_path, dde_corpus_query_path,field='dde')

##########
### CS ###
##########
'''
cs_paper_json_path= os.path.join(raw_path,"data/CS_dic_new.json")#,os.path.join(raw_path,"data/CS_dic_1017.json")
cs_index_path = os.path.join(raw_path,"output/cs/cs_dde_sim")
cs_corpus_query_path = os.path.join(raw_path,"output/cs_corpus_query.json")
cs_paper_dict, wiki_tiny_dict, cs_index, cs_corpus_query = get_preload(cs_paper_json_path, cs_index_path, cs_corpus_query_path, wiki_json_path, field='cs')
'''
@app.post("/api/v1/acekg/paperee")
async def paper_qa_entity_extraction(pid: str, field='DDE'):#(pid: str, field:str)
    paper_dict = dde_paper_dict
    index = dde_index
    corpus_query = dde_corpus_query
    if field=='CS':
        paper_dict = cs_paper_dict
        index = cs_index
        corpus_query = cs_corpus_query
    paperkg_final = get_paperkg_final(paper_dict, index, corpus_query, pid, field,abs_w=0.2, title_w=0.3, mr_w=0.5, word_len_w=1.0, letter_len_w=0.01, complex_len_w=1.0)
    return paperkg_final

#uvicorn.run(app='main:app', host="0.0.0.0", port=8012, reload=False, debug=False)