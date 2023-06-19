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

#device = torch.device("cuda")
print ("====================================================================================================================")
#print ("using gpu: ", torch.cuda.get_device_name(1))
#torch.cuda.set_device(2)
#print(torch.cuda.current_device())


torch.cuda.empty_cache()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')


model_name = 'digitalepidemiologylab/covid-twitter-bert-v2'

bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

#model_name = 'bert-large-uncased'
#bert = BertModel.from_pretrained('bert-large-uncased')

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

        #softmax activation function
        #self.softmax = nn.LogSoftmax(dim=1)
        #self.sigmoid = nn.Sigmoid(dim=1)
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
        

        # apply softmax activation
        #x = self.softmax(x)
        #x = self.sigmoid(x)
        
        return x

    
    
    
def get_data(df, label,  feat='TWEET_TEXT_PROCESSED', test_labels=[]):
    
    
    #now_time = time.time()
    #print ('==============time to initialize tokenizer')
    #print (now_time-start_time)

    #seq_len = [len(i.split()) for i in train_text]

    #pd.Series(seq_len).hist(bins = 30)

    print (df[feat].shape)
    
    
    #start_time = time.time()
    
    
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

def T_scaling(logits, temperature):
    #temperature = args.get('temperature', None)
    return torch.div(logits, temperature)



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
    #start_time = time.time()
    
    model = BERT_Arch(bert)
    #now_time = time.time()
    #print ('==============time to initialize model')
    #print (now_time-start_time)
    
    #start_time = time.time()
    model = nn.DataParallel(model).cuda()
    #model = nn.DataParallel(model)
    # push the model to GPU
    
    
    device = torch.device('cuda')
    #device = torch.device('cpu')
    #now_time = time.time()
    
    #print ('==============time to store model on gpu')
    #print (now_time-start_time)
    
    #load weights of best model
    #path = '/home/adhiman/SAR-z/classifier/ct_bert/saved_weights_ctbertdnn_2_auth.pt'
    #path = '/home/adhiman/SAR-z/classifier/ct_bert/saved_weights_ctbertdnn_2_fam.pt'
    #start_time = time.time()
    
    #state_dict = clean_state(model_path)
    #model.load_state_dict(state_dict)
    model.load_state_dict(torch.load(model_path, map_location=device )) # , map_location=device
    
    #now_time = time.time()
    #print ('==============time to load model')
    #print (now_time-start_time)
    #start_time = time.time()
    
    
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
        

    #print (total_preds_.shape)
    #print (total_preds_)
    #pred_lbl = np.argmax(total_preds_, axis = 1)
    #print (pred_lbl.shape)
    
    #T_pred_lbl = np.argmax(total_T_preds, axis = 1)

    
    torch.cuda.empty_cache()
    
    '''
    
    twt_seq = twt_seq.to(device)
    twt_mask = twt_mask.to(device)
    #test_y = test_y.to(device)
    #print (model.is_cuda)
    print (twt_seq.is_cuda)
    print (twt_mask.is_cuda)
    #model = model.to(device)
    #now_time = time.time()
    #print ('==============time to move to device')
    #print (now_time-start_time)
    
    
    #start_time = time.time()
    
    # get predictions for test data
    with torch.no_grad():
        preds = model(twt_seq, twt_mask)
        preds_ = preds.detach().cpu().numpy()
    
    #now_time = time.time()
    #print ('==============time to get predictions')
    #print (now_time-start_time)
    
    #start_time = time.time()
    
    #print (preds)
    pred_lbl = np.argmax(preds_, axis = 1)
    
    if T!='x':
        T_pred = T_scaling(preds, T)
    
    T_pred = T_pred.detach().cpu().numpy()
    
    T_pred_lbl = np.argmax(T_pred, axis = 1)
    #print (preds)
    #now_time = time.time()
    #print ('==============time to get labels')
    del preds
    
    #print (now_time-start_time)
    torch.cuda.empty_cache() 
    
    print(classification_report(y_test, pred_lbl))

    print ("================================ORIGINAL==========================================================")
    print (precision_score(y_test, pred_lbl,pos_label=0, average ='macro'))
    print (recall_score(y_test, pred_lbl,pos_label=0, average ='macro'))
    print (f1_score(y_test, pred_lbl,pos_label=0, average ='macro'))
    print (accuracy_score(y_test, pred_lbl))
    print (confusion_matrix(y_test, pred_lbl, labels=[1, 0]))

    print ("=====major class report======")
    all_false = [False]*len(y_test)
    #print(classification_report(all_false, clas))
    print (precision_score(all_false, pred_lbl,pos_label=0))
    print (recall_score(all_false, pred_lbl,pos_label=0))
    print (f1_score(all_false, pred_lbl,pos_label=0))
    print (accuracy_score(all_false, pred_lbl))
    print (confusion_matrix(all_false, pred_lbl, labels=[1, 0]))

    print ("=====minor class report======")
    all_tru = [True]*len(y_test)
    #print(classification_report(all_tru))
    print ("==========================================================================================")
    print (precision_score(all_tru, pred_lbl,pos_label=1))
    print (recall_score(all_tru, pred_lbl,pos_label=1))
    print (f1_score(all_tru, pred_lbl,pos_label=1))
    print (accuracy_score(all_tru, pred_lbl))
    print (confusion_matrix(all_tru, pred_lbl, labels=[1, 0]))

    return preds_, pred_lbl, T_pred, T_pred_lbl'''
    #return total_preds_, pred_lbl, total_T_preds, T_pred_lbl
    return total_preds_, total_T_preds
    


def getinput1(label):
    
    #start_time = time.time()
    #df = pd.read_csv('/disks/sdb/adhiman/SAR-z/labelbox itr3/without consesus/matching/test.csv')
    #df = pd.read_csv('/disks/sdb/adhiman/SAR-z/labelbox itr3/with consensus/data_for_validation/itr123_test.csv')
    df = read_pickle1('/disks/sdb/adhiman/SAR-z/labelbox itr3/with consensus/data_for_validation/itr123.pkl')
    
    #df =df[:100]
    print (df.shape)
    df['ABT_FAMILY'] = df['FAMILY'].astype(bool)
    df = clean_tweets(df, False, remove_duplicates=False)
    print ("after preprocessing")
    print (df.shape)
    df = df.replace('NO', 0).replace('YES', 1)
    
    df["AUTHOR_OR"] = df["AUTHOR_COVID"] + df["AUTHOR_SYMPTOMS"] > 0
    df["FAMILY_OR"] =df["FAMILY_COVID"] + df["FAMILY_SYMPTOMS"] > 0
    
    '''class_size = df[label].sum()
    df1 = pd.concat([
    df[df[label] == 0].sample(int(class_size)),
    df[df[label] == 1]])
    
    df =df1'''
    
    #df[label] = list(map(lambda ele: ele == "True", df[label].tolist()))
    
    #print (df.head())
    print (df.shape)
    
    test_labels = df[label]
    #now_time = time.time()
    
    #print ('==============time1')
    #print (now_time-start_time)
    #df = clean_tweets(df, remove_duplicates=True)
    print ("================")
    #print (df.head())
    print (df.shape)
    return df, test_labels



    
    

        
def main_test():
    batch_size = 1024
    label = 'FAMILY_OR'
    feat = 'TWEET_TEXT_PROCESSED'
    
    #df = read_pickle1('/disks/sdb/adhiman/SAR-z/uk_bow_vectors/1231175993248899073/api.pkl')
    #df = read_pickle1('/home/adhiman/SAR-z/Eda/processed/labeled_1_2_augmented.pkl')
    #df = read_pickle1('/home/adhiman/SAR-z/Eda/processed/new/auged_1_2.pkl')
    #df= read_pickle1('/home/adhiman/SAR-z/Eda/processed/new/train-val-test/test.pkl')
    
    df, test_labels = getinput1(label)
    
    
        
    if label =='AUTHOR_OR':
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/saved_weights_ctbertdnn_2_auth2.pt'
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/ctbert/saved_weights_ctbertdnn_auth_test5.pt'
        load_path ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/bert/saved_weights_bert_auth_test5.pt'
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/saved_weights_bert_auth_test.pt'
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/final_/with last layer tuned/saved_weights_ctbertdnn_2_auth2.pt'
    if label =='FAMILY_OR':
        #load_path = '/home/adhiman/SAR-z/classifier/ct_bert/saved_weights_ctbertdnn_2_fam2.pt'
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/ctbert/saved_weights_ctbertdnn_fam_test5.pt'
        load_path = '/home/adhiman/SAR-z/classifier/ct_bert/testing weight/bert/saved_weights_bert_fam_test5.pt'
        #load_path= '/home/adhiman/SAR-z/classifier/ct_bert/saved_weights_bert_fam_test.pt'
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/final_/with last layer tuned/saved_weights_ctbertdnn_2_fam2.pt'
        
    
    #start_time = time.time()
    #bert, twt_seq, twt_mask, twt_y = get_data(df,label,test_labels , feat)
    twt_seq, twt_mask, twt_y  = get_data(df,label, feat,test_labels )
    
    #now_time = time.time()
    
    #print ('==============time2')
    #print (now_time-start_time)
    
    
    #start_time = time.time()
    
    if label =='AUTHOR_OR':
        pred, p_lbl, T_pred, T_p_lbl = test(twt_seq, twt_mask, batch_size, load_path,2.79, twt_y)
    if label =='FAMILY_OR':
        pred, p_lbl, T_pred, T_p_lbl = test( twt_seq, twt_mask,  batch_size, load_path, 3.53, twt_y)
    
    #now_time = time.time()
    
    #print ('==============time3')
    #print (now_time-start_time)
    
    #start_time = time.time()
    
    test_df = pd.DataFrame(pd.np.column_stack([df[['TWEET_ID', 'TWEET_TEXT','TWEET_TEXT_PROCESSED', label]] , pred])).rename(columns={0: 'TWEET_ID'})
    
    #print (test.shape)
    test_df = pd.DataFrame(pd.np.column_stack([test_df , T_pred]))
    
    #print (test.shape)
    
    test_df['Predicted Labels'] = p_lbl
    test_df['T_Predicted Labels'] = T_p_lbl

    #now_time = time.time()
    
    #print ('==============time4')
    #print (now_time-start_time)
    
    #print (auth_scr)
    print (test_df.head())
    test_df=test_df.rename(columns={0: 'TWEET_ID', 1: 'TWEET_TEXT', 2: 'TWEET_TEXT_PROCESSED', 3:'Actual Label', 4: 'PRED_0', 5: 'PRED_1', 6: 'T_PRED_0', 7: 'T_PRED_1'})
    test_df.to_csv(f"/home/adhiman/SAR-z/classifier/"+label+"all_ct_preds.csv", index=None)

    
def main_prob():
    batch_size = 3072
    #label = 'AUTHOR_OR'
    feat = 'TWEET_TEXT_PROCESSED'
    print (feat)
    #all_users = glob.glob('/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/*/api.pkl')
    
    pr = pd.read_csv('/home/adhiman/SAR-z/classifier/p_rall.csv')
    #print (pr.head())
    users = pr[pr['0']=='fail']['Unnamed: 0'].tolist()
    
    all_users = [''.join(['/disks/sdb/adhiman/SAR-z/uk_bow_vectors/', str(x),'/api.pkl' ]) for x in users ]
    
    print (all_users[:5])
    
    
    print ("====Accessing users: ",len(all_users))
    
    
    '''if label =='AUTHOR_OR':
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/final_/with last layer tuned/saved_weights_ctbertdnn_2_auth2.pt'
        load_path ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/saved_weights_ctbertdnn_auth_test100.pt'
    if label =='FAMILY_OR':
        #load_path ='/home/adhiman/SAR-z/classifier/ct_bert/final_/with last layer tuned/saved_weights_ctbertdnn_2_fam2.pt'
        load_path = '/home/adhiman/SAR-z/classifier/ct_bert/testing weight/saved_weights_ctbertdnn_fam_test100.pt'
            '''
    
    #load_path_auth ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/saved_weights_ctbertdnn_auth_test100.pt'
    #load_path_fam = '/home/adhiman/SAR-z/classifier/ct_bert/testing weight/saved_weights_ctbertdnn_fam_test100.pt'
    
    load_path_auth = '/disks/sdb/adhiman/SAR-z/ct_bert/testing weight/final_wts/ctbertdnn_auth_test1.pt'
    load_path_fam = '/disks/sdb/adhiman/SAR-z/ct_bert/testing weight/final_wts/ctbertdnn_fam_test1.pt'
    load_path_abtfam = '/home/adhiman/SAR-z/classifier/ct_bert/testing weight/final_wts/ctbertdnn_abt_fam101.pt'
    
    
    label_list = ['AUTHOR_OR', 'FAMILY_OR', 'ABT_FAMILY']
    for i in tqdm(range(0, len(all_users), 1000)):
        users = all_users[i:i+1000]
        #print (users)
        df_temp = pd.DataFrame()
        to_remove = []
        for u in users:
            user  = u.split('/')[-2]
            '''if label=='AUTHOR_OR' and os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors/'+str(user)+'/auth_ct_score3.pkl'):
                #print ('going back')
                continue
    
            if label=='FAMILY_OR' and os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors/'+str(user)+'/fam_ct_score3.pkl'):
                #print ('going back')
                continue'''
            
            '''if os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/'+str(user)+'/auth_ct_score4.pkl') and os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/'+str(user)+'/fam_ct_score4.pkl') and os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/'+str(user)+'/abtfam_ct_score4.pkl') :
                continue
            label_list = []
            if not os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/'+str(user)+'/auth_ct_score4.pkl'):
                label_list.append('AUTHOR_OR')
            if not os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/'+str(user)+'/fam_ct_score4.pkl'):
                label_list.append('FAMILY_OR')
            if not os.path.exists('/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/'+str(user)+'/abtfam_ct_score4.pkl'):
                label_list.append('ABT_FAMILY')'''
            
            
            df_f = read_pickle1(u)
            if df_f.shape[1]!=0:
                df_temp =pd.concat([df_temp, read_pickle1(u)])
            else:
                to_remove(u)
                
        if df_temp.shape[0]!=0:
            users =[item for item in users if item not in to_remove]
            print ("tweets dont exist for users")
            print (len(to_remove))
            #print (df_temp.shape)
            #print (df_temp.head())
            #print (df_temp.columns)
            df_temp = clean_tweets(df_temp, False, remove_duplicates=False)
            #print (df_temp.shape)
            
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

                '''test_df['Predicted Labels'] = p_lbl
                test_df['Temp Predicted Labels'] = T_p_lbl'''

                #now_time = time.time()

                #print ('==============time4')
                #print (now_time-start_time)

                #print (test_df.head())
                #print (test_df.columns)

                for u in users:
                    user  = u.split('/')[-2]
                    #print (user)
                    df_u = test_df[test_df['USER_ID']==user]
                    
                    if label=='FAMILY_OR':
                        df_u.drop(columns =['USER_ID','TWEET_TEXT_PROCESSED']).set_index('TWEET_ID').to_pickle('/disks/sdb/adhiman/SAR-z/uk_bow_vectors/'+str(user)+'/fam_ct_score4.pkl')
                    if label=='AUTHOR_OR':
                         df_u.drop(columns =['USER_ID','TWEET_TEXT_PROCESSED']).set_index('TWEET_ID').to_pickle('/disks/sdb/adhiman/SAR-z/uk_bow_vectors/'+str(user)+'/auth_ct_score4.pkl')
                    if label=='ABT_FAMILY':
                         df_u.drop(columns =['USER_ID','TWEET_TEXT_PROCESSED']).set_index('TWEET_ID').to_pickle('/disks/sdb/adhiman/SAR-z/uk_bow_vectors/'+str(user)+'/abtfam_ct_score4.pkl')
                            
                #test.to_csv(f"/home/adhiman/SAR-z/classifier/"+label+"test_ct_preds.csv", index=None)




        

if __name__ == "__main__":
    
    
    #main_test()
    main_prob()