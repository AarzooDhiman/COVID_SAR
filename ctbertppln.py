import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import logging
import time
from platform import python_version
import pickle5 as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
#from torch.autograd import Variable
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from transformers import AdamW
from sklearn import preprocessing
from transformers import (
   AutoConfig,
   AutoModel,
   AutoTokenizer,
   TFAutoModelForSequenceClassification,
   AdamW,
   glue_convert_examples_to_features
)
import time
import glob
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
 # optimizer from hugging face transformers
from transformers import AdamW

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import transformers
from transformers import AutoModel, BertTokenizerFast
#import tensorflow_datasets as tfds
from numba import cuda

from sklearn.preprocessing import OneHotEncoder
#device = cuda.get_current_device()
#device.reset()
from sadice import SelfAdjDiceLoss

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from preprocess import clean_tweets
from transformers import BertTokenizer, BertModel

print (torch.__version__)

print ("====================================================================================================================")

torch.cuda.empty_cache()

model_name = 'digitalepidemiologylab/covid-twitter-bert-v2'

bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def read_pickle1(pth):
    with open(pth, "rb") as fh:
        data = pickle.load(fh)
    return (data)

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
      
        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        #self.fc1 = nn.Linear(768,512)
        self.fc1 = nn.Linear(1024,512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,1)

        self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        

        return x

    
def get_data(df, label,  feat='TWEET_TEXT_PROCESSED', test_labels=[]):
   
    
    
    # tokenize and encode sequences in the training set
    twt_tokens = tokenizer.batch_encode_plus(
        df[feat].tolist(),
        max_length = 60, #96
        pad_to_max_length=True,
        truncation=True
    )


    ## convert lists to tensors

    twt_seq = torch.tensor(twt_tokens['input_ids'])
    twt_mask = torch.tensor(twt_tokens['attention_mask'])
    if len(test_labels)!=0:
        twt_y = torch.tensor(test_labels.tolist())
        #now_time = time.time()
        #print ('==============time to get tokens')
        #print (now_time-start_time)

        return ( twt_seq, twt_mask, twt_y)
    else:
        return ( twt_seq, twt_mask)



def clean_state(path):
    device = torch.device('cpu')
    state_dict = torch.load(path, map_location=device)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    
    return new_state_dict
    #model.load_state_dict(new_state_dict)
    
    
def test( twt_seq, twt_mask,  batch_size, model_path, T='x', y_test=[]):
    
    
    model = BERT_Arch(bert)
    
    model = nn.DataParallel(model).cuda()
    #model = nn.DataParallel(model)
    # push the model to GPU
    
    
    device = torch.device('cuda')
    
    
    #load weights of best model
    #path = 'saved_weights_ctbertdnn_2_auth.pt'
    #path = 'saved_weights_ctbertdnn_2_fam.pt'
    
    
    model.load_state_dict(torch.load(model_path, map_location=device )) # , map_location=device
    
   
    twt_data = TensorDataset(twt_seq, twt_mask)

    # sampler for sampling the data during training
    twt_sampler = SequentialSampler(twt_data)

    # dataLoader for validation set
    twt_dataloader = DataLoader(twt_data, sampler = twt_sampler, batch_size=batch_size)

    
    total_T_preds = []
    total_preds_ = []
    
    
    for runen, batch in enumerate(twt_dataloader):
        # print (runen)
        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            preds_ = preds.detach().cpu().numpy()
            
            pred_sigmoid = torch.sigmoid(torch.from_numpy(preds_))
            
            if T!='x':
                T_pred = T_scaling(preds, T)
                
            T_pred = T_pred.detach().cpu().numpy()
                
            
            #total_preds_ =total_preds_ + list(preds_)
            total_preds_ =total_preds_ + list(pred_sigmoid)
            total_T_preds =total_T_preds + list(T_pred)
            
            
    total_preds_ = np.array(total_preds_)    
    total_T_preds = np.array(total_T_preds)    
        
    
    torch.cuda.empty_cache()
    
    return total_preds_, total_T_preds
    


def getinput1(label):
    

    df = read_pickle1('/labeled_tweets.pkl')

    df['ABT_FAMILY'] = df['FAMILY'].astype(bool)
    df = clean_tweets(df, False, remove_duplicates=False)

    df = df.replace('NO', 0).replace('YES', 1)
    
    df["AUTHOR_OR"] = df["AUTHOR_COVID"] + df["AUTHOR_SYMPTOMS"] > 0
    df["FAMILY_OR"] =df["FAMILY_COVID"] + df["FAMILY_SYMPTOMS"] > 0
    
    print (df.shape)
    
    test_labels = df[label]

    return df, test_labels
        
def main_test():
    batch_size = 1024
    label = 'FAMILY_OR'
    feat = 'TWEET_TEXT_PROCESSED'
    
    
    df, test_labels = getinput1(label)
        
    if label =='AUTHOR_OR':
        load_path ='saved_weights_bert_auth_test5.pt'
       
    if label =='FAMILY_OR':
        load_path = 'saved_weights_bert_fam_test5.pt'
        
    twt_seq, twt_mask, twt_y  = get_data(df,label, feat,test_labels )

    if label =='AUTHOR_OR':
        pred, p_lbl, T_pred, T_p_lbl = test(twt_seq, twt_mask, batch_size, load_path,2.79, twt_y)
    if label =='FAMILY_OR':
        pred, p_lbl, T_pred, T_p_lbl = test( twt_seq, twt_mask,  batch_size, load_path, 3.53, twt_y)
    
    
    test_df = pd.DataFrame(pd.np.column_stack([df[['TWEET_ID', 'TWEET_TEXT','TWEET_TEXT_PROCESSED', label]] , pred])).rename(columns={0: 'TWEET_ID'})
    
    test_df = pd.DataFrame(pd.np.column_stack([test_df , T_pred]))
    
    test_df['Predicted Labels'] = p_lbl
    test_df['T_Predicted Labels'] = T_p_lbl

 
    print (test_df.head())
    test_df=test_df.rename(columns={0: 'TWEET_ID', 1: 'TWEET_TEXT', 2: 'TWEET_TEXT_PROCESSED', 3:'Actual Label', 4: 'PRED_0', 5: 'PRED_1', 6: 'T_PRED_0', 7: 'T_PRED_1'})
    test_df.to_csv(f"{label}_all_ct_preds.csv", index=None)

    
def main_prob():
    batch_size = 3072
    
    feat = 'TWEET_TEXT_PROCESSED'
    print (feat)
    
    pr = pd.read_csv('/home/adhiman/SAR-z/classifier/p_rall.csv')
    #print (pr.head())
    users = pr[pr['0']=='fail']['Unnamed: 0'].tolist()
    
    all_users = [''.join(['uk_users/', str(x),'/api.pkl' ]) for x in users ]
    
    print (all_users[:5])
    
    print ("====Accessing users: ",len(all_users))
    

    load_path_auth = 'ctbertdnn_auth_test1.pt'
    load_path_fam = 'ctbertdnn_fam_test1.pt'
    load_path_abtfam = 'ctbertdnn_abt_fam101.pt'
    
    
    label_list = ['AUTHOR_OR', 'FAMILY_OR', 'ABT_FAMILY']
    for i in tqdm(range(0, len(all_users), 1000)):
        users = all_users[i:i+1000]
        #print (users)
        df_temp = pd.DataFrame()
        to_remove = []
        for u in users:
            user  = u.split('/')[-2]
            
            df_f = read_pickle1(u)
            if df_f.shape[1]!=0:
                df_temp =pd.concat([df_temp, read_pickle1(u)])
            else:
                to_remove(u)
                
        if df_temp.shape[0]!=0:
            users =[item for item in users if item not in to_remove]
            print ("tweets dont exist for users")
            print (len(to_remove))
            
            df_temp = clean_tweets(df_temp, False, remove_duplicates=False)
            
            
            for label in label_list:
                print ("DOING -----", label)
                twt_seq, twt_mask = get_data(df_temp,label, feat,test_labels=[] )
                if label =='AUTHOR_OR':
                    #pred, p_lbl, T_pred, T_p_lbl = test(twt_seq, twt_mask, batch_size,  load_path_auth,1, []) #3.530
                    pred, T_pred = test(twt_seq, twt_mask, batch_size,  load_path_auth,1, []) #3.530
                    
                if label =='FAMILY_OR':
                    #pred, p_lbl, T_pred, T_p_lbl = test( twt_seq, twt_mask,batch_size, load_path_fam, 1 , []) #1.897
                    pred,  T_pred = test( twt_seq, twt_mask,batch_size, load_path_fam, 1 , []) #1.897
                    
                if label =='ABT_FAMILY':
                    #pred, p_lbl, T_pred, T_p_lbl = test( twt_seq, twt_mask,batch_size, load_path_fam, 1 , []) #1.897
                    pred,  T_pred = test( twt_seq, twt_mask,batch_size, load_path_abtfam, 1 , []) #1.897
                    

                test_df = pd.DataFrame(pd.np.column_stack([df_temp[['TWEET_ID','USER_ID', 'TWEET_TEXT','TWEET_TEXT_PROCESSED']] , pred])).rename(columns={0: 'TWEET_ID', 1: 'USER_ID', 2: 'TWEET_TEXT', 3: 'TWEET_TEXT_PROCESSED', 4: 'PRED'})

                print (test_df.shape)
                test_df = pd.DataFrame(pd.np.column_stack([test_df , T_pred])).rename(columns={0: 'TWEET_ID', 1: 'USER_ID', 2: 'TWEET_TEXT', 3: 'TWEET_TEXT_PROCESSED', 4: 'PRED', 5: 'T_PRED'})
                test_df['PRED'] = test_df['PRED'].apply(lambda x: x.numpy()[0])


                for u in users:
                    user  = u.split('/')[-2]
                    #print (user)
                    df_u = test_df[test_df['USER_ID']==user]
                    
                    if label=='FAMILY_OR':
                        df_u.drop(columns =['USER_ID','TWEET_TEXT_PROCESSED']).set_index('TWEET_ID').to_pickle('/uk_user_scores/'+str(user)+'/fam_ct_score4.pkl')
                    if label=='AUTHOR_OR':
                         df_u.drop(columns =['USER_ID','TWEET_TEXT_PROCESSED']).set_index('TWEET_ID').to_pickle('/uk_user_scores/'+str(user)+'/auth_ct_score4.pkl')
                    if label=='ABT_FAMILY':
                         df_u.drop(columns =['USER_ID','TWEET_TEXT_PROCESSED']).set_index('TWEET_ID').to_pickle('/uk_user_scores/'+str(user)+'/abtfam_ct_score4.pkl')
                            

if __name__ == "__main__":
    
    
    #main_test()
    main_prob()
