import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
import pandas as pd
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import time
import numpy as np
import pickle
import os
from tqdm.auto import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

def add_multi_index(lst):
    return list(range(lst))

def select_longer_claims(df):
    
    #to_del = df[(df.type == "train") & (df.source_text_sentences_len <= min_no_sentence_source_text)].id.to_list()
    #df = df[~ df.id.isin(to_del)]
    df = df.explode(['source_text_sentences',"source_text_sentences_index"])
    
    return df

def split_data(dat, sour_col, targ_col):  
    
    dat["label"] = list(range(len(dat)))
    dat['source_text'] = dat[sour_col]
    dat['target_text'] = dat[targ_col]

    splitter = SentenceSplitter(language='en')
    dat['source_text_sentences'] = dat['source_text'].apply(lambda x : splitter.split(text = x))
    dat['source_text_sentences_len'] = dat['source_text_sentences'].str.len()
    dat['source_text_sentences_index'] = dat['source_text_sentences_len'].apply(lambda x : add_multi_index(x))

    #dat = dat[["id","source_text", "target_text", "source_text_sentences", "source_text_sentences_len","source_text_sentences_index","type"]]
    return dat


def embeddings_sentence_bert(text, IsBase, Bert_name):  
    
        start = time.time()
        if IsBase==True:            
            model = SentenceTransformer(Bert_name, device = 'cuda:0')  # model  bert-base-uncased           
        else:     
                     
            word_embedding_model = models.Transformer(Bert_name)
            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = 'cuda:0')

        
        #Sentences are encoded by calling model.encode()
        sentence_vectors = model.encode(text,show_progress_bar=True, batch_size = 500)            

        end = time.time()

        print("Time for creating "+ str(len(sentence_vectors))+" embedding vectors " + str((end - start)/60))
        print('Model used :'+ Bert_name )

        return sentence_vectors
    
def get_similarity_matrix(df,metric = "cosine"):
     
    df = df.to_list()    
    A =  np.array(df,dtype=float)
    A_sparse = sparse.csr_matrix(A)

    if (metric=="cosine"):
        similarities = cosine_similarity(A_sparse)
        similarities_norm = (1/(1+similarities))                #      (1/(1+similarities))
    elif(metric=="euclidean"):
        similarities = euclidean_distances(A_sparse)
        similarities_norm= (1/(1+similarities))
    return np.mean(similarities_norm, axis=0)

def get_lof_score(dat):
    ad = []
    try:
        lof = LocalOutlierFactor(n_neighbors = neighbours,metric='cosine')
        embeds = dat.to_list()
        lof.fit_predict(embeds).tolist()
        return 1 - (1/(1-(lof.negative_outlier_factor_)))
    except:
        return [0.51]*len(dat)

def create_embeddings(df):
    sentences_lst = df["source_text_sentences"].tolist() #answer
    embeddings = embeddings_sentence_bert(sentences_lst, True, base_model_name)
    df["source_text_sentences_embed_base"] =  embeddings.tolist()
    return df