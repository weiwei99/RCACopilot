import fasttext
import numpy as np
import pandas as pd
import cufflinks as cf
from transformers import AutoTokenizer
import warnings
cf.set_config_file(offline=True)
warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
from utils import trainingFile_preprocess

"""
This file is for Fasttext Model
"""

df = pd.read_csv('cri_head.csv', lines=True)
df=[['Title','RootCauseId','Summary','Mitigation']]
# all_keywords = df['Keyword'].unique().tolist()
df['metadata']=df.to_dict(orient="records")
train = df
train = train.sort_values(by=['AlertId'])
# use ICL from train to construct the test dataset by recalling samples in the training dataset

file=trainingFile_preprocess(train)
np.savetxt('Data_Models/train_tem.txt',file,fmt='%s')
model = fasttext.train_supervised(input='Data_Models/train_tem.txt',epoch=200,lr=0.9,thread=7,wordNgrams=2,seed=41)
model.save_model('Data_Models/models')

