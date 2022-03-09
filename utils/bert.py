import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from tqdm.auto import tqdm
import pickle
import itertools
from sklearn import metrics
import operator
import warnings
from sentence_transformers import models, losses
import time
from datetime import datetime
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from sentence_splitter import SentenceSplitter
import time
import torch, gc
import math
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
tqdm.pandas()

def get_triple_from_df(df, idx_col ,
                       label_col, text_col,
                       claim_col):
    """
    Function to to create triples from main text column by target label
    """

   
    triplets = []
    negatives_lst = []
    labels = df[label_col].unique()
    
    
    #idxs = df[idx_col].to_list()
    #sentences = df[text_col].to_list()
    #claims = df[claim_col].to_list()
    
    for label in tqdm(labels):
        data_in = df[(df[label_col] == label)][[text_col,claim_col]]
        data_out = df[(df[label_col] != label)]
        data_len = len(data_in)
        data_out_indexes = data_out.index
        numbers = np.random.choice(data_out_indexes, data_len)
        negatives = data_out.loc[numbers,text_col].to_list()
        data_in["negatives"] = negatives
        triplets.append(data_in)
    res = pd.concat(triplets)
    res = [InputExample(texts = [str(rows[claim_col]), str(rows[text_col]), str(rows["negatives"])]) for index, rows in tqdm(res.iterrows())]
    return res

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
        sentence_vectors = model.encode(text, show_progress_bar=True, batch_size = 1000)            

        end = time.time()

        print("Time for creating "+ str(len(sentence_vectors))+" embedding vectors " + str((end - start)/60))
        print('Model used :'+ Bert_name )

        return sentence_vectors

def add_multi_index(lst):
    return list(range(lst))

def split_data(dat, sour_col):  
    
    dat["label"] = list(range(len(dat)))
    dat['source_text'] = dat[sour_col]

    splitter = SentenceSplitter(language='en')
    dat['source_text_sentences'] = dat['source_text'].progress_apply(lambda x : splitter.split(text = x))
    dat['source_text_sentences_len'] = dat['source_text_sentences'].str.len() 
    dat['source_text_sentences_index'] = dat['source_text_sentences_len'].progress_apply(lambda x : add_multi_index(x))
    dat = dat.explode(['source_text_sentences',"source_text_sentences_index"]).reset_index()
    dat['sentence_len'] = dat['source_text_sentences'].str.split().str.len()
    
    dat = dat[dat.sentence_len > 5]

    #dat = dat[["id","source_text", "target_text", "source_text_sentences", "source_text_sentences_len","source_text_sentences_index","type"]]
    return dat