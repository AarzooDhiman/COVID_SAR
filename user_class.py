# -*- coding: utf-8 -*-


import pickle5 as pickle
import pandas as pd
from multiprocessing import Pool
from multiprocessing import Process
import threading
import time
import glob, os, json, time
import calendar
import numpy as np
import matplotlib.pyplot as plt  
import datetime
import sys
import itertools
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import random
from collections import defaultdict
import re
from scipy import stats
import pytz

tau_a = -0.63
tau_f = 0.028
max_a = 2.662
max_f = 2.610
min_a = -6.162
min_f = -6.575

def read_pickle1(pth):
    with open(pth, "rb") as fh:
        data = pickle.load(fh)
    return (data)


emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002500-\U00002BEF"  # chinese char
                       u"\U00002702-\U000027B0"
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\U00010000-\U0010ffff"
                       u"\u2640-\u2642"
                       u"\u2600-\u2B55"
                       u"\u200d"
                       u"\u23cf"
                       u"\u23e9"
                       u"\u231a"
                       u"\ufe0f"  # dingbats
                       u"\u3030"
                       "]+", flags=re.UNICODE)

'''raw_tweets = []
for h in glob.glob('/disks/sda/adhiman/SAR-z/raw_tweets/monthly_files/*.pkl'):
    raw_tweets = raw_tweets + read_pickle1(h)['TWEET_ID'].tolist()

raw_tweets = list(set(raw_tweets))

print ("RAW TWEETS")
print (len(raw_tweets))
'''


def find_best_pair(auth_df, fam_df, days):
    
    auth_df1 = auth_df[['TWEET_ID', 'USER_ID', 'AUTHOR_SCORE' , 'MONTH_ID', 'TIMESTAMP', 'PROB_A']]
    auth_df1= auth_df1.rename(columns={'TWEET_ID':'AUTH_TWEET_ID', 'MONTH_ID':'AUTH_MONTH_ID', 'TIMESTAMP':'AUTH_TIMESTAMP' })
    fam_df1 = fam_df[['TWEET_ID', 'USER_ID', 'FAMILY_SCORE' , 'MONTH_ID', 'TIMESTAMP', 'PROB_F']]
    fam_df1= fam_df1.rename(columns={'TWEET_ID':'FAM_TWEET_ID', 'MONTH_ID':'FAM_MONTH_ID', 'TIMESTAMP':'FAM_TIMESTAMP' })
    
    #auth_df1['PROB_A'] =  0.5 + 0.5*((auth_df1['AUTHOR_SCORE']- tau_a)/(max_a - tau_a))
    #fam_df1['PROB_F'] =  0.5 + 0.5*((fam_df1['FAMILY_SCORE']- tau_f)/(max_f - tau_f))
    

    
    auth_df1 = auth_df1.sort_values(['AUTHOR_SCORE'], ascending=False) ##finding maximum
    fam_df1 = fam_df1.sort_values(['FAMILY_SCORE'], ascending=False) ##finding maximum
    
    #auth_df1.to_csv('/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/test/33192983_dfa_top10_og.csv')
    #fam_df1.to_csv('/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/test/33192983_dff_top10_og.csv') 

    
    #auth_df1 = auth_df1.head(5)
    #fam_df1 = fam_df1.head(5)
    
    '''print (auth_df1.columns)
    print (auth_df1.shape)
    print (fam_df1.columns)
    print (fam_df1.shape)'''
    
    df_all = auth_df1.merge(fam_df1, how='outer')
    
    #df_all.to_csv('/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/test/33192983_top10_og.csv')
    
    
        
    #df_all = df_all[df_all['AUTH_TWEET_ID']!=df_all['FAM_TWEET_ID']]
    #df_all = df_all.sort_values(['AUTHOR_SCORE', 'FAMILY_SCORE'], ascending=False).reset_index() ##finding maximum
    
    #print ("df_cross join")
    #print (df_all)
    #print (df_all.shape)
    df_all['delta'] = (df_all['FAM_TIMESTAMP'] - df_all['AUTH_TIMESTAMP']).dt.days
    df_all['PROB'] = df_all['PROB_A']*df_all['PROB_F']
    df_all = df_all.sort_values(['PROB'], ascending=False).reset_index()
    #print (df_all['delta'].values[0])
    
    
    #print (df_all[['AUTH_TIMESTAMP','FAM_TIMESTAMP', 'delta']])

    #print ("************************")
    #print (df_all.head())
    df_all1 = df_all[(df_all['delta']>=(0-days)) & (df_all['delta']<=days) & (df_all['delta']!=0)]
    #df_all1.to_csv('/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/test/33192983_top10_filt_og.csv')
    
     ##finding maximum
    '''if df_all1.shape[0]!=0:
        print ("-----------------3")
        print (df_all1.head())'''
    
    #print (df_all[['AUTH_TIMESTAMP','FAM_TIMESTAMP','AUTHOR_SCORE','FAMILY_SCORE', 'delta']])
    
    #time_interval={}
    
    if df_all1.shape[0]>0:
        final_delta =df_all1['delta'].values[0]
        neg_values = df_all1[df_all1['delta']<0].shape[0]
        pos_values = df_all1[df_all1['delta']>0].shape[0]
        
        
        #print ("delta")
        #print (df_all1['delta'].values[0])
        '''if neg_values>0 and pos_values>0:
            return (0, -1,-1, {}, 0 )
        else:
            final_auth_mnth = df_all1['AUTH_MONTH_ID'].values[0]
            final_fam_mnth = df_all1['FAM_MONTH_ID'].values[0]
            
            time_interval[final_auth_mnth] = final_delta
            return (final_delta, final_auth_mnth, final_fam_mnth, time_interval, df_all1['PROB'].values[0] )'''
        final_auth_mnth = df_all1['AUTH_MONTH_ID'].values[0]
        final_fam_mnth = df_all1['FAM_MONTH_ID'].values[0]

        #time_interval[final_auth_mnth] = final_delta
        return (final_delta, final_auth_mnth, final_fam_mnth, df_all1['PROB_A'].values[0] , df_all1['PROB_F'].values[0] )
    
    elif df_all1.shape[0]==0:
        if df_all.shape[0]>0:
            '''if (df_all['delta'].values[0] >14) and (df_all['delta'].values[0]<30):
                print (df_all['delta'].values[0])'''
            #time_interval[df_all['AUTH_MONTH_ID'].values[0]] = df_all['delta'].values[0]
            return (df_all['delta'].values[0], df_all['AUTH_MONTH_ID'].values[0], df_all['FAM_MONTH_ID'].values[0], df_all['PROB_A'].values[0] , df_all['PROB_F'].values[0] )
        else:
            return "nothing"
        
        '''elif df_all.shape[0]==0:
            print (df_all)
            return (df_all['delta'].values[0], auth_df1['AUTH_MONTH_ID'].values[0],fam_df1['FAM_MONTH_ID'].values[0] )'''
            
        '''p_a = 0.5 + 0.5*((auth_df1['AUTHOR_SCORE'].values[0]- tau_a)/(max_a - tau_a))
            p_f = 0.5 + 0.5*((fam_df1['FAMILY_SCORE'].values[0]- tau_f)/(max_f - tau_f))
            
            #print ("data not found")
            #print (auth_df1['AUTHOR_SCORE'])
            #print (fam_df1['FAMILY_SCORE'])
            #print ( p_a, p_f)
            
            if p_a>p_f:
                return (10000, auth_df1['AUTH_MONTH_ID'].values[0], -1 )
            elif p_f>p_a:
                return (10000, -1, fam_df1['FAM_MONTH_ID'].values[0] )
            else:
                return (0, -1,-1 )'''
            
            

def classes(author_positive_df, family_positive_df, fam_cnf, auth_cnf, path, user):
    
    author_positive = author_positive_df.copy()
    family_positive = family_positive_df.copy()
    #family_any = family_any_df.copy()
    
    
    classification = {}
    serial_interval = {}
    
    #print (author_positive)
    #print (family_positive)
    
    #start_of_timeline = datetime.datetime(2020, 1, 1, tzinfo=pytz.UTC)
    
    if author_positive.shape[0]>0:
        #print (author_positive)
        
        
        author_positive.TIMESTAMP = pd.to_datetime(author_positive.TIMESTAMP, utc=True)
       
        #print (author_positive.TIMESTAMP)
        
        #print ("auth shape")
        #print (author_positive)
        
        
        
        author_positive["MONTH_ID"] = author_positive.TIMESTAMP.dt.month + (author_positive.TIMESTAMP.dt.year-2020)*12 ##mnth
        
        '''author_positive['delta'] = author_positive.TIMESTAMP - start_of_timeline ##week
        author_positive['week'] = (author_positive['delta'].dt.days // 7) + 1
        author_positive['MONTH_ID'] = author_positive['week']'''
        #print ("auth shape")
        #print (author_positive)
        
        #author_positive = author_positive.reset_index().iloc[0] #finding maximum ###update
        #auth_month_id = int(author_positive.MONTH_ID) ###update
        author_positive= author_positive.rename(columns={'log':'AUTHOR_SCORE'})
        
        author_positive = author_positive.sort_values("AUTHOR_SCORE", ascending=False).reset_index() ##finding maximum
        
        author_positive['PROB_A'] =  0.5 + 0.5*((author_positive['AUTHOR_SCORE']- tau_a)/(max_a - tau_a))
        '''if author_positive.shape[0]>10:
            author_positive = pd.DataFrame()'''
            
        #author_positive = author_positive[:21]

        
    if family_positive.shape[0]>0:
        #print (family_positive)
        family_positive.TIMESTAMP = pd.to_datetime(family_positive.TIMESTAMP, utc=True)
        
        #print ("fam shape")
        #print (family_positive.shape[0])
        
        family_positive["MONTH_ID"] = family_positive.TIMESTAMP.dt.month + (family_positive.TIMESTAMP.dt.year-2020)*12 #mnth
        
        '''family_positive['delta'] = family_positive.TIMESTAMP - start_of_timeline ##week
        family_positive['week'] = (family_positive['delta'].dt.days // 7) + 1
        family_positive['MONTH_ID'] = family_positive['week']'''
        
        
        #print (family_positive)
        #family_positive= family_positive.rename(columns={0:'FAMILY_SCORE'})
        #family_positive= family_positive.rename(columns={'T_PRED':'FAMILY_SCORE'})
        family_positive= family_positive.rename(columns={'log':'FAMILY_SCORE'})
        
        #print (family_positive)
        
        #family_positive = family_positive.sort_values("FAMILY_SCORE", ascending=False).reset_index().iloc[0] ##finding maximum
        family_positive = family_positive.sort_values("FAMILY_SCORE", ascending=False).reset_index()
        
        family_positive['PROB_F'] =  0.5 + 0.5*((family_positive['FAMILY_SCORE']- tau_f)/(max_f - tau_f))
        
        #fam_month_id  = int(family_positive.MONTH_ID) ####update
        #if fam_month_id ==10:
            #print ("---------------family")
            #print (family_positive.TWEET_TEXT)
            
            
        '''if family_positive.shape[0]>10:
            family_positive = pd.DataFrame()'''
        #family_positive = family_positive[:21]
            
    
    wd={}
    #print (author_positive)
    #print (family_positive)
    
    class_p=0
    
    class_p={}
    
    month_ids = {'auth_month':[], 'fam_month':[]}
    
    #if (author_positive.shape[0]<=5) and  (family_positive.shape[0]<=5):
    
    
    if author_positive.shape[0]>0 and family_positive.shape[0]==0:

        auth_month_id = int(author_positive.iloc[0]['MONTH_ID'])
        #print ("author")
        #print (author_positive['TWEET_TEXT'].values[0])

        classification[auth_month_id]=str(10)
        wd[auth_month_id] = auth_cnf
        p_val = author_positive.iloc[0]['PROB_A']
        month_ids['auth_month'] = author_positive['MONTH_ID'].tolist()
        class_p[auth_month_id] = p_val

    if author_positive.shape[0]==0 and family_positive.shape[0]>0:
        fam_month_id = int(family_positive.iloc[0]['MONTH_ID'])

        #print ("family")
        #print (family_positive['TWEET_TEXT'].values[0])

        classification[fam_month_id]=str(20)
        wd[fam_month_id] = fam_cnf
        p_val = family_positive.iloc[0]['PROB_F']
        month_ids['fam_month'] = family_positive['MONTH_ID'].tolist()
        class_p[fam_month_id] = p_val

    if author_positive.shape[0]>0 and family_positive.shape[0]>0:

        days = 14
        delta, auth_month_id, fam_month_id, p_a, p_f = find_best_pair(author_positive, family_positive, days)
        
        #print (type(delta))
        #print (delta)

        month_ids['auth_month'] = author_positive['MONTH_ID'].tolist()
        month_ids['fam_month'] = family_positive['MONTH_ID'].tolist()


        #delta = pd.Timedelta(family_positive.TIMESTAMP-author_positive.TIMESTAMP).days



        if (delta>=1) and (delta<=days):
            #print ('12')
            classification[auth_month_id] = str(12)
            wd[auth_month_id] = (fam_cnf +auth_cnf)/2
            serial_interval[auth_month_id] = delta
            p_val = (p_a+p_f)/2
            class_p[auth_month_id] = p_val


        elif delta>=(0-days) and (delta<=-1):
            #print ('21')
            classification[fam_month_id] = str(21)
            wd[fam_month_id] = (fam_cnf +auth_cnf)/2
            serial_interval[fam_month_id] = delta
            p_val = (p_a+p_f)/2
            class_p[fam_month_id] = p_val

        elif delta!=0:                                           #####correct this 

            #print ('1020')

            '''if '10' in usr_cls:
                classification[auth_month_id] = str(10)
            if '20' in usr_cls:
                classification[fam_month_id] = str(20)'''
            #print (auth_month_id)
            #print (fam_month_id)
            classification[auth_month_id] = str(10)
            wd[auth_month_id] = auth_cnf
            class_p[auth_month_id] = p_a
            
            classification[fam_month_id] = str(20)
            wd[fam_month_id] = fam_cnf
            class_p[fam_month_id] = p_a

            '''classification[fam_month_id] = str(21020)
            classification[auth_month_id] = str(11020)'''

            #serial_interval[auth_month_id] = delta


        else:
            serial_interval = {}
            pass
        




    json_cls = {str(k):v for k,v in classification.items() if k>=0}
    classification = {k:v for k,v in classification.items() if k>=0}
    wd_sc = {k:v for k,v in wd.items() if k>=0}
    
    #print (serial_interval)
    if len(classification) > 0:
        with open(f"{path}/classifications.json", 'w') as fp:
            json.dump(json_cls, fp) 
        with open(f"{path}/months.json", 'w') as fm:
            json.dump(month_ids, fm)
        
            
            
    return classification, wd_sc, serial_interval, class_p
            

    
def author_fam_positive(user, path, out_path, auth_th, fmth_th):
    path1 =''
    auth_conf = 0
    fam_conf = 0
    exist=True
    if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/author_positive.csv'):
        author_positive = pd.read_csv(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/author_positive.csv')
        #print (author_positive)
        
        if author_positive.shape[0]>0:
            path1 =out_path+str(user)
            try:
                auth_conf = author_positive['PRED'].values[0]
            except:
                print ("exception 1")
                print (user)
                #print (author_positive)
        else:
            auth_conf = min_a
            
        if not os.path.exists(out_path+str(user)):
                os.mkdir(out_path+str(user))
    else:
        author_positive =pd.DataFrame()

    if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/family_positive.csv'):
        family_positive= pd.read_csv(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/family_positive.csv')
        
        if family_positive.shape[0]>0:
            path1 =out_path+str(user)
            fam_conf = family_positive['PRED'].values[0]
        else:
            fam_conf = min_f

        if not os.path.exists(out_path+str(user)):
                os.mkdir(out_path+str(user))
                
    else:
        family_positive =pd.DataFrame()
        
    if os.path.exists(f'{path}{user}/abtfam_ct_score4.pkl'):
        abt_fam = read_pickle1(f'{path}{user}/abtfam_ct_score4.pkl')
        if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/abtfam_ct_score4.pkl'):
            abt_new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/abtfam_ct_score4.pkl')
            abt_fam  = pd.concat([abt_fam,abt_new_twts])
        if abt_fam.shape[0]>0:
            p_f = abt_fam[abt_fam['T_PRED']>2].shape[0]/abt_fam.shape[0]
            #print (abt_fam.head())
            abt_fam_twts= abt_fam[abt_fam['T_PRED']>2].index.tolist()
            
        else:
            p_f = 0

    if len(path1)>0:
        class_assg, eld_w, serial_interval, class_p = classes(author_positive, family_positive, fam_conf, auth_conf,  path1, user)
        if len(class_assg)>0:
            #print (user)
            #print (class_assg)
            return (user,eld_w, p_f, class_assg, serial_interval, class_p)

    
        
def author_fam_positive2(user, path, out_path, auth_th, fmth_th):
    
    thresh= {}
    thresh['author_threshold']= auth_th
    thresh['family_threshold']= fmth_th
    

    try:
        tweets = read_pickle1(f'{path}{user}/api.pkl')
        if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/api.pkl'):
            new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/api.pkl')
            tweets  = pd.concat([tweets,new_twts])
    except:
        return
   
    
    if os.path.exists(f'{path}{user}/auth_ct_score4.pkl'):
        author_tweets = read_pickle1(f'{path}{user}/auth_ct_score4.pkl')
        if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/auth_ct_score4.pkl'):
            a_new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/auth_ct_score4.pkl')
            author_tweets  = pd.concat([author_tweets,a_new_twts])
            
        
        author_tweets['log'] = np.log(author_tweets['T_PRED'].astype('float32'))
        author_tweets= author_tweets.fillna(-100)
        author_tweets = author_tweets[['log', 'PRED', 'T_PRED']]
        
        #print (author_tweets)
        #author_tweets = author_tweets.squeeze('columns') 
        
        #print (author_tweets)
    else:
        author_tweets = pd.DataFrame()
    
    if os.path.exists(f'{path}{user}/fam_ct_score4.pkl'):
        family_tweets = read_pickle1(f'{path}{user}/fam_ct_score4.pkl')
        if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/fam_ct_score4.pkl'):
            f_new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/fam_ct_score4.pkl')
            family_tweets  = pd.concat([family_tweets,f_new_twts])
        
        family_tweets['log'] = np.log(family_tweets['T_PRED'].astype('float32'))
        family_tweets= family_tweets.fillna(-100)
        family_tweets = family_tweets[['log', 'PRED', 'T_PRED']]
        
        #family_tweets = family_tweets.squeeze('columns')
        #print (family_tweets)
    else:
        family_tweets = pd.DataFrame()
        
    
    tweets = tweets.set_index("TWEET_ID")
    
    tweets.index = tweets.index.map(int)

    tweets.TIMESTAMP = pd.to_datetime(tweets.TIMESTAMP, utc=True)

    author_tweets = author_tweets[author_tweets.index.astype('str').isin(tweets[tweets.TIMESTAMP > "2020-01-01"].index.astype('str'))]

    family_tweets = family_tweets[family_tweets.index.astype('str').isin(tweets[tweets.TIMESTAMP > "2020-01-01"].index.astype('str'))]

   
    tweets = tweets.reset_index()
    
    path1 =''
    tweets['USER_ID'] =  tweets['USER_ID'].astype('str')
    tweets['TWEET_ID'] =  tweets['TWEET_ID'].astype('str')
    
    

    #print (author_tweets.shape)
    #print (family_tweets.shape)
    
    author_positive =pd.DataFrame()
    family_positive =pd.DataFrame()
    
    #print ("here")

    
    if author_tweets.shape[0]>0:

        #max_score =author_tweets.max()  
        #max_author = str(author_tweets.idxmax())## with new classifier
        #max_author = author_tweets[author_tweets==max_score].index[0]
        #print ('maxscore' +str(max_score) +str(thresh['author_threshold']))
        #print (str(user) + out_path)
        
        auth_max = author_tweets[author_tweets['log'] >= thresh['author_threshold']] ## with new classifier
        
        #print (auth_max)
  
        auth_max = auth_max.reset_index()
        
        #if max_score > thresh['author_threshold']:
        if auth_max.shape[0]>0:
            #print ('here')
            if not os.path.exists(out_path+str(user)):
                os.mkdir(out_path+str(user))

    if family_tweets.shape[0]>0:
        '''family_max = family_tweets[family_tweets.FAMILY_SCORE >= thresh['family_threshold']]'''
        family_max = family_tweets[family_tweets['log'] >= thresh['family_threshold']] ## with new classifier
        
        #print (family_max)
        
        family_max = family_max.reset_index()
        #print ('max'+str(family_max))
        if family_max.shape[0] > 0:
            #print ('here')
            if not os.path.exists(out_path+str(user)):
                os.mkdir(out_path+str(user))
   
    auth_conf = 0
    fam_conf = 0
    exist=True

    if author_tweets.shape[0]>0:
        #if max_score > thresh['author_threshold']:
        if auth_max.shape[0] > 0:
            
            #print ("THIS HAPPENING")
            #author_max = author_tweets.loc[author_tweets.index.astype(str)==max_author]

            #author_max = author_max.reset_index()
            auth_max['TWEET_ID'] = auth_max['TWEET_ID'].astype('str')


            author_positive = auth_max.merge(tweets, on='TWEET_ID',how='inner' )
            #print (author_positive)
            

            path1 =out_path+str(user)
            
            author_positive = author_positive.drop_duplicates(subset=['TWEET_ID'])
            author_positive['TWEET_TEXT'] = author_positive['TWEET_TEXT'].apply(lambda x: re.sub('@[A-Za-z0-9_]+','', x))
            author_positive['TWEET_TEXT'] = author_positive['TWEET_TEXT'].apply(lambda x: re.sub('http\S+','', x))
            author_positive['TWEET_TEXT'] = author_positive['TWEET_TEXT'].apply(lambda x: re.sub('www.\S+','', x))
            author_positive['TWEET_TEXT'] = author_positive['TWEET_TEXT'].apply(lambda x: emoji_pattern.sub(r'', x))
            author_positive['TWEET_TEXT'] = author_positive['TWEET_TEXT'].apply(lambda x: x.strip())
            author_positive = author_positive[author_positive['TWEET_TEXT']!='']
            author_positive = author_positive.drop_duplicates(subset=['TWEET_TEXT'])
             
            author_positive = author_positive.dropna(subset=['log'])
            
            #print (author_positive)
            if author_positive.shape[0]<=98: ###removing the outlier case (just one)
                if not os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/'):
                    os.mkdir(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/')
                #print (author_positive)
                author_positive.to_csv(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/author_positive.csv')
            else:
                author_positive = pd.DataFrame()
                #print ("auth")
            #print (author_positive)
            '''auth_twt = read_pickle1(f'{path}{user}/auth_ct_score4.pkl')
            if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/auth_ct_score4.pkl'):
                a_new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/auth_ct_score4.pkl')
                auth_twt  = pd.concat([auth_twt,a_new_twts])
            author_positive['TWEET_ID'] = author_positive['TWEET_ID'].astype('str')
            auth_twt  =auth_twt.reset_index()
            auth_df= author_positive.merge(auth_twt, on ='TWEET_ID')
            print (auth_df)'''
            
            if author_positive.shape[0]>0:
                auth_conf = author_positive['PRED'].values[0]
            else:
                auth_conf = min_a
                
        else:
            exist = False
            
            

    if os.path.exists(f'{path}{user}/abtfam_ct_score4.pkl'):
        abt_fam = read_pickle1(f'{path}{user}/abtfam_ct_score4.pkl')
        if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/abtfam_ct_score4.pkl'):
            abt_new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/abtfam_ct_score4.pkl')
            abt_fam  = pd.concat([abt_fam,abt_new_twts])
        if abt_fam.shape[0]>0:
            p_f = abt_fam[abt_fam['T_PRED']>2].shape[0]/abt_fam.shape[0]
            #print (abt_fam.head())
            abt_fam_twts= abt_fam[abt_fam['T_PRED']>2].index.tolist()
            
        else:
            p_f = 0
    

    if family_tweets.shape[0]>0:
        if family_max.shape[0] > 0:

            family_max['TWEET_ID'] = family_max['TWEET_ID'].astype('str')

            family_positive = family_max.merge(tweets, on='TWEET_ID', how='inner')
            
            if (family_positive.shape[0]==0):
                print ("=======================================")
                print (user)
                print ("=======================================")
            
            path1 =out_path+str(user)
            #if (family_positive[family_positive['FAMILY_SCORE'] == family_positive.loc[family_positive['FAMILY_SCORE'].idxmax(), 'FAMILY_SCORE']].shape[0]>3):
                #print (user)
            #print ('---')
            #print (family_positive.shape)
            
            family_positive = family_positive[family_positive['TWEET_ID'].isin(abt_fam_twts)]
            #print (family_positive.shape)
            
            
            family_positive = family_positive.drop_duplicates(subset=['TWEET_ID'])
            family_positive['TWEET_TEXT'] = family_positive['TWEET_TEXT'].apply(lambda x: re.sub('@[A-Za-z0-9_]+','', x))
            family_positive['TWEET_TEXT'] = family_positive['TWEET_TEXT'].apply(lambda x: re.sub('http\S+','', x))
            family_positive['TWEET_TEXT'] = family_positive['TWEET_TEXT'].apply(lambda x: re.sub('www.\S+','', x))
            family_positive['TWEET_TEXT'] = family_positive['TWEET_TEXT'].apply(lambda x: emoji_pattern.sub(r'', x))
            family_positive['TWEET_TEXT'] = family_positive['TWEET_TEXT'].apply(lambda x: x.strip())
            family_positive = family_positive[family_positive['TWEET_TEXT']!='']
            family_positive = family_positive.drop_duplicates(subset=['TWEET_TEXT'])
            family_positive = family_positive.dropna(subset= ['log'])
            
            
            if family_positive.shape[0]<=75: ####removing the outlier case (just one)
                if not os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/'):
                    os.mkdir(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/')

                #print (family_positive)
                family_positive.to_csv(f'/disks/sdb/adhiman/SAR-z/uk_ctbert_positives/{user}/family_positive.csv')
                #print ("fam")
                #print (family_positive)

                #print (family_positive.columns)
            else:
                family_positive = pd.DataFrame()

            
            
            #family_positive.to_csv(f"{path1}/family_positive.csv")
            
            '''fam_twt = read_pickle1(f'{path}{user}/fam_ct_score4.pkl')
            if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/fam_ct_score4.pkl'):
                f_new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/fam_ct_score4.pkl')
                x="here goes"
                fam_twt  = pd.concat([fam_twt,f_new_twts])
            #family_positive = family_positive.sort_values("log", ascending=False).reset_index().head(1)
            family_positive = family_positive.sort_values("log", ascending=False).reset_index()
            family_positive['TWEET_ID'] = family_positive['TWEET_ID'].astype('str')
            fam_twt  =fam_twt.reset_index()
            fam_df= family_positive.merge(fam_twt, on ='TWEET_ID')'''
            
            
            if family_positive.shape[0]>0:
                fam_conf = family_positive['PRED'].values[0]
          
            else:
                fam_conf = min_f
                
        else:
            exist = False
    
    if len(path1)>0:
        #class_assg = classes(author_positive, family_positive, family_any, path1, user)
        class_assg, eld_w, serial_interval, class_p = classes(author_positive, family_positive, fam_conf, auth_conf,  path1, user)
        if len(class_assg)>0:
            #print (user)
            #print (class_assg)
            return (user,eld_w, p_f, class_assg, serial_interval, class_p)

        

    
    
def calSAR(classes0):
    classes = classes0.copy()
    #print (classes)
    final_classes = {}
    pr_f = {}
    w={}
    si = {}
    all_probs = {}
    for vec in classes:

        try:
            final_classes[vec[0]] = vec[3]
            pr_f[vec[0]] = vec[2]
            w[vec[0]] = vec[1]
            si[vec[0]] = vec[4]
            #all_probs.append(vec[5])
            all_probs[vec[0]]=vec[5]
        except:
            
            pass
            #count =count+1
            #print (vec)
        #print (classes)
        #print (vec)
        #exit()
    
    #print (final_classes)
    
    #print (w)
    #print (count)
    summary = pd.DataFrame.from_dict(final_classes, orient="index")
    rate = pd.DataFrame.from_dict(pr_f, orient="index", columns = ['RATE'])
    si_df = pd.DataFrame.from_dict(si, orient="index")
    #si_df = si_df.abs()
    #print (summary)
    #print (rate)
    #print (si_df)
    si_df.to_csv('/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/shifting_classes/top_tweets/user_si_p0.csv')
    
    si_df = si_df.apply(lambda x:x.mean(), axis =0)
    si_df = si_df.to_frame().rename(columns= {0:'si'})
    #print (si_df)
    
    w_df = pd.DataFrame.from_dict(w, orient="index")
    w_df = w_df.reset_index()
    w_df = w_df.rename(columns ={'index':'user'})
    #print (w_df)
    summary= summary.sort_index(axis=1)
    
    prob_sum = pd.DataFrame( columns=['10', '20', '12', '21'], index = list(range(29)))
    
    prob_sum = prob_sum.replace(np.nan, 0.0)
    
    
    for pair, clss in final_classes.items():
        for m, c  in clss.items():
            try:
                prob_sum.iloc[int(m)-1][c] =prob_sum.loc[int(m)-1][c] + all_probs[pair][m]
            except:
                print ("exception 2")
                print (pair)
                print (m)
                print (all_probs[pair][m])
                
    prob_sum['fSAR'] = (prob_sum['12']+prob_sum['21'])/(prob_sum['12']+prob_sum['21']+prob_sum['10']+prob_sum['20'])          
    #prob_sum.to_csv('/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/shifting_classes/14_prob_top10.csv')
    
    summary = summary.merge(rate,left_index=True, right_index=True)
    #summary["RATE"] = 1
    #print (summary)
    return summary, w_df, si_df, all_probs
    
    #elad_res = elad_sar(summary)
    #print ("elad_res")
    #print (elad_res)

    
def elad_sar2(summ, w_df):
    summary = summ.copy()
    #print (summary.head())
    #print (summary[1].value_counts())
    #print (summary.columns.values)
    elad_SARs = {}
    #print (summary)


    #print (w_df.head())

    #summary2 = summary.merge(w_df[month_ID], left_on=summary.index, right_on='user')
    #print (summary2)
    #summary2['RATE'] =0.5
    #summary2['RATE'] = np.random.rand(summary2.shape[0])*0.1
    #print (summary.head())
    #print (summary2.head())
    
    psar = {}
    c_val =list(summary.columns.values)
    c_val.remove('RATE')
    
    #w_df.to_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/user_w.csv', index =False)
    #w_13 =  w_df[['user', 13]].dropna(subset=[13])
    #w_13.to_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/user_w_13.csv', index =False)
    
    p_x = summary[summary['RATE']!=0.0].shape[0]/77016
    
    print ("px--------------")
    print (p_x)
    
    for month_ID in c_val:
        #print ("+++++++++++++++++++++++++++++++++++++++++")
        m_u =pd.DataFrame(summary[month_ID]).rename(columns={month_ID:'T'}).dropna(subset=['T'])
        #print (w_df.columns.values)
        #print (month_ID)
        w_u = w_df[['user',month_ID]].rename(columns={month_ID:'conf'})
        #print(w_u[['user','conf']])
        summary2 = summary.merge(w_u, left_on=summary.index, right_on='user')
        #print (summary2)
        #print (summary2.head())
        mnth_sum = summary2.merge(m_u, left_on='user', right_on =m_u.index )
        
        
        #print (m_u)
        #print (m_u.shape)
        
        #mnth_sum = summary2.dropna(subset=[month_ID])
        #print (mnth_sum)
        #print (mnth_sum.shape)
        
            
        
        #T_R= mnth_sum['T'].replace('12', 1.0).replace('21', 1.0).replace('10', 0.0).replace('20',0.0)
        T_R= mnth_sum['T'].replace('12', 1.0).replace('21', 1.0).replace('10', 0.0).replace('20',0.0)###update
        
        T_R = T_R.fillna(0.0)
        T_R = T_R.astype('float64')
        #print (T_R.value_counts())
        T_R = T_R.to_numpy()
        #print (T)
        #P_R = mnth_sum['RATE'].to_numpy()
        #print (P_R)
        #P_R = np.array([1.0]*mnth_sum.shape[0])
        
        P_R = np.array([p_x]*mnth_sum.shape[0])
        
        #P_R = np.array([0.87065]*mnth_sum.shape[0])
        W = np.diag(mnth_sum.conf)

        p_left = np.matmul(np.matmul(P_R, W), P_R.T)
        #print (p_left)
        p_right = np.matmul(np.matmul(P_R,W), T_R.T)
        #print (p_right)
        #ps = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(P_R, W), P_R.T)),P_R),W),T_R.T)
        psar[month_ID] = p_right/p_left
        
            
        
        #psar[month_ID] = ps
    Elad_SAR_df = pd.DataFrame.from_dict(psar, orient="index").rename({0: "new_Elad_SAR"}, axis=1)
    #print (Elad_SAR_df)
    return Elad_SAR_df

        
def elad_sar(summ):
    summary = summ.copy()
    print (summary)
    #summary.to_csv('/disks/sdb/adhiman/SAR-z/temp/sum.csv')
    elad_SARs = {}
    #print (summary)
    summary['RATE'] =1
    for month_ID in summary.columns.values:
        #print ("+++++++++++++++++++++++++++++++++++++++++")
        P_R = pd.concat([summary[(summary[month_ID] == '12') | (summary[month_ID] == '21')].RATE, summary[summary[month_ID] == '10'].RATE]).to_numpy()
        #print (P_R)
        P_R =  np.array([P_R]).T
        #print (P_R)
        T = np.concatenate([np.ones(summary[(summary[month_ID] == '12') | (summary[month_ID] == '21')].shape[0]), np.zeros(summary[summary[month_ID] == '10'].shape[0])])
        #print (T)
        T = np.array([T]).T
        #print (T.shape)
        P_SAR  = np.matmul(
            np.matmul(
                    np.linalg.pinv(
                            np.matmul(P_R.T,P_R)),P_R.T), T)
        #print (P_SAR)
        elad_SARs[month_ID] = P_SAR[0][0]
        
    Elad_SAR_df = pd.DataFrame.from_dict(elad_SARs, orient="index").rename({0: "Elad_SAR"}, axis=1)
    return Elad_SAR_df
    
    
def pred_sar(summ,elad, gold_df, si, total_users):
    
    summary = summ.copy()
    #print (summary.columns)
    #print (elad)
    #summary.to_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/summary.csv')
    #summary = summary.drop(columns=['RATE'])
    
    
    p_x = summary[summary['RATE']!=0.0].shape[0]/77016
    
    print ("old px")
    print (p_x)
    
    #p_x = summary[summary['RATE']!=0.0].shape[0]/62061
    #p_x = summary[summary['RATE']!=0.0].shape[0]/total_users
    #p_x = float(summary.shape[0]/77016)
    #print ("___________________________")
    #print (summary.shape[0])
    #print (p_x)
    class_summ = summary.drop(columns=['RATE']).apply(pd.Series.value_counts).T
    
    #print (summary)
    print ('==================================')
    #print (p_x)
    p= {}
    p_2 ={}
    p_df = pd.DataFrame()
    for x in summary.drop(columns= ['RATE']).columns.values:
        #y = summary[[x,'RATE']].replace('10', np.nan).replace('20', np.nan)
        y = summary[[x,'RATE']].replace('10', np.nan).replace('20', np.nan).replace('11020', np.nan).replace('21020', np.nan) ###update
        
        y = y.dropna()
        #print (summary[[x]].count().values[0])
        c = summary[[x]].count().values[0]
        p_u = y['RATE'].sum()/y.shape[0]
        p[x] = p_u
        #print (summary[[x]])
        valid_users = summary[[x]].dropna().index.tolist()
        #print (valid_users)
        y2 = summary[summary.index.isin(valid_users)][[x,'RATE']]
        #print (y2)
        #p_u2 = y2[y2['RATE']>0.005].shape[0]/len(valid_users)
        p_u2 = y2[y2['RATE']>0.0].shape[0]/summary.shape[0]  ###separate userbase ??
        #print (p_u2)
        p_2[x] = p_u2

        '''y = summary[[x,'RATE']].dropna()
        p_u = y['RATE'].mean()
        p[x] = p_u'''

    #print (p)
    
    
    p_df = pd.DataFrame.from_dict(p, orient='index', columns = ['P_U'])
    
    p_df2 = pd.DataFrame.from_dict(p_2, orient='index', columns = ['P_U2'])
    #print (p_df2)
    #p_df2.to_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/p_df2.csv')
    #print (si)
    class_summ =class_summ.merge(si, left_index =True, right_index=True )
    #print (class_summ)
    
    class_summ =class_summ.merge(p_df2, left_index =True, right_index=True )
    class_summ = class_summ.merge(p_df, left_index = True, right_index = True)
    #print (class_summ)
    
    
    #print (class_summ.shape)
    table =class_summ.copy()
    table =table.fillna(0)
    #print (table)
    #for cl in ['10','20','12','21']:
    for cl in ['10','20','12','21']: ###update
        if cl not in table.columns.values:
            table[cl]=0
    table["A2"] = table['21']+table['20']
    table["A1"] = table['12']+table['10']
    table["A"] = table['12']+table['10']+table['21']+table['20']
    #table["A"] = table['12']+table['10']+table['21']+table['20']+table['1020']  ####can't use that because A10intA20!=0 ####update
    table["S1"] = (table['12']+table['10'])/(table['12']+table['10']+table['21']+table['20'])
    
    table["S2"] = (table['21']+table['20'])/(table['12']+table['10']+table['21']+table['20'])
    
    #table["S1"] = (table['12']+table['10'])/(table['12']+table['10']+table['21']+table['20']+table['1020']) ###update
    #table["S2"] = (table['21']+table['20'])/(table['12']+table['10']+table['21']+table['20']+table['1020']) ### update
    
    #table['new_SAR'] = ((table["A2"]/table["A"])*table["S2"]) + ((table["A1"]/table["A"])*table["S1"])
    #table['P_SAR'] = ((table['P_U']*table['21'])+table['12'])/(table['A']) ###update
    #table['alpha2'] = (table['21'])/(table['21']+table['20'])  ##update
    #table['alpha1'] = (table['12'])/(table['12']+table['10']) ##update
    table['alpha2'] = (table['21'])/(table['21']+table['20'])
    table['alpha1'] = (table['12'])/(table['12']+table['10'])
    '''table['a1/a2'] = table['alpha1']/table['alpha2']
    
    p_x = np.mean(table['a1/a2'])
    print ("alpha1 alpha2 mean")
    print (p_x)'''
    
    
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(table['alpha1'].tolist(), table['alpha2'].tolist())
    print ("======slope")
    print (slope)
    print (intercept)
    
    #table['p_alpha1'] = (table['12'])/(p_x*(table['12']+table['10']))##older method of mean(alpha1/alpha2)
    table['p_alpha1'] = slope*table['alpha1']+intercept
    
    #table['p_alpha2'] = (table['12'])/(table['P_U2']*(table['12']+table['10']))
    
    table['fSAR'] = table["S1"]*table['alpha1']+table["S2"]*table['alpha2']
    table['pSAR'] = table["S1"]*table['p_alpha1']+table["S2"]*table['alpha2']
    #table['pSAR2'] = table["S1"]*table['p_alpha2']+table["S2"]*table['alpha2']
    
    
    #table["SAR_sum"] = table["S1"] + table["S2"]
    #table["SAR_avg"] = (table["S1"] + table["S2"])/2
    #table["SAR_wtd"] = (table["S1"]*(table['12']+table['10']) + table["S2"]*(table['21']+table['20']))/(table['12']+table['10']+table['21']+table['20'])
    #alternative SAR calculations
    table["NAIVE_SAR"] = (table['12'] + table['21']) / table["A"]
    #print (table)
    table = table.join(elad)
    #print (table)
    table = table.join(gold_df)
    #print (table)
    datetime_series = pd.Series(pd.date_range("2019-12-01", periods=table.shape[0]+1, freq="M")) #+int(table.index[0]) ###mnth    
    
    datetime_series = datetime_series.dt.strftime("%b-%y")[table.index[0]:] ###mnth

    
    #datetime_series = pd.Series(pd.date_range("2020-01-01", periods=table.shape[0]+1, freq="W")) ##week
    #datetime_series =datetime_series+datetime.timedelta(days=2) #week
    #datetime_series = datetime_series.dt.strftime("%d-%b-%y")[table.index[0]:] ###week
    #datetime_series.index += 1 
    print (datetime_series)
    
    table = table.merge(datetime_series.to_frame(), left_index=True, right_index=True) ##mnth
    #table['Week'] = datetime_series.dt.strftime("%d-%b-%y") ##week
    table = table.rename(columns={0:'Month'}) ##mnth
    
    #table.index = datetime_series
    print (table)
    
    #table.dropna(inplace =True)
    #print (table)
    return table
    #return table.iloc[3:-1]
    
    
def gold():
    gold2min = 0.08
    gold2max = 0.48
    gold1min = 0.046
    gold1max = 0.37
    gold3min = 0.113
    gold3max = 0.53

    gold_months = {
        0:{"gold_min": gold1min,
       "gold_max": gold1max},
        1:{"gold_min": gold1min,
       "gold_max": gold1max},
        2:{"gold_min": gold1min,
       "gold_max": gold1max},
        3:{"gold_min": gold1min,
       "gold_max": gold1max},
        4:{"gold_min": gold1min,
       "gold_max": gold1max},
        5:{"gold_min": gold1min,
       "gold_max": gold1max},
        6:{"gold_min": gold1min,
       "gold_max": gold1max},
        7:{"gold_min": gold1min,
       "gold_max": gold1max},
        8:{"gold_min": gold2min,
       "gold_max": gold2max},
        9:{"gold_min": gold2min,
       "gold_max": gold2max},
        10:{"gold_min": gold2min,
       "gold_max": gold2max},
        11:{"gold_min": gold2min,
       "gold_max": gold2max},
        12:{"gold_min": gold2min,
       "gold_max": gold2max},
        13:{"gold_min": gold3min,
       "gold_max": gold3max},
        14:{"gold_min": gold3min,
       "gold_max": gold3max},
        15:{"gold_min": gold3min,
       "gold_max": gold3max},
        16:{"gold_min": gold3min,
       "gold_max": gold3max},
        17:{"gold_min": gold3min,
       "gold_max": gold3max},
        18:{"gold_min": gold3min,
       "gold_max": gold3max},
        19:{"gold_min": gold3min,
       "gold_max": gold3max},
        20:{"gold_min": gold3min,
       "gold_max": gold3max},
        21:{"gold_min": gold3min,
       "gold_max": gold3max},
        22:{"gold_min": gold3min,
       "gold_max": gold3max},
        23:{"gold_min": gold3min,
       "gold_max": gold3max},
        24:{"gold_min": gold3min,
       "gold_max": gold3max}
    }
    gold_df = pd.DataFrame.from_dict(gold_months, orient="index")
    return gold_df


def prep_ground():
    '''#gr = pd.read_csv("/disks/sda/adhiman/SAR-z/ground_truth/phe_sar_household.csv", index_col='Week_commencing')
    gr = pd.read_csv("/disks/sda/adhiman/SAR-z/ground_truth/phe_sar_household_new.csv", index_col='Week_commencing')
    #gr = pd.read_csv("/disks/sda/adhiman/SAR-z/ground_truth/phe_sar_household-withoutajdust.csv", index_col='Week_commencing')
    #print (gr)
    gr.index = pd.to_datetime(gr.index, format="%d/%m/%Y")
    #print (gr)
    gr['alpha_perc'] = gr['alpha_cases']/(gr['alpha_cases']+gr['delta_cases'])
    gr['delta_perc'] = gr['delta_cases']/(gr['alpha_cases']+gr['delta_cases'])
    gr['PHE_SAR'] = gr['alpha_perc']*gr['alpha_sar'] + gr['delta_perc']*gr['delta_sar']
    gr['PHE_SAR'] = gr['PHE_SAR']/100
    
    #print (gr)
    gr = gr.resample('M').mean()
    gr.index = gr.index.strftime("%b-%y")
    #print (gr)
    users = pd.read_csv('/disks/sda/adhiman/SAR-z/ground_truth/monthly_users.csv')
    users['Month'] =pd.to_datetime(users['Month'], format="%b-%y") 
    users['Month'] = users['Month'].dt.strftime("%b-%y")
    #print (users)
    gr =gr.merge(users, left_on =gr.index, right_on = 'Month', how='outer')
    #print (gr)'''
    
    gr = pd.read_csv('/disks/sda/adhiman/SAR-z/ground_truth/phe_sar_only.csv')
    gr.index = pd.to_datetime(gr.Month, format="%b %y")
    gr.index = gr.index.strftime("%b-%y")
    #print (gr)
    
    users = pd.read_csv('/disks/sda/adhiman/SAR-z/ground_truth/monthly_users.csv')
    users['Month'] =pd.to_datetime(users['Month'], format="%b-%y") 
    users['Month'] = users['Month'].dt.strftime("%b-%y")
    #gr =gr.merge(users, left_on =gr.index, right_on = 'Month', how='outer') ####mnth
    #print (gr)
    return gr

def plot_data(tab, all_probs,  path, ath,fth,v, r):
    table =tab.copy()
    ath =round(ath,3)
    fth =round(fth,3)
    #print (table)
    print ("using number of users", v)
    
    #df_probs = pd.DataFrame(all_probs, columns=["p"])
    #df_probs.to_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/validation/prob_distr_all.csv')
    
    #start_date = datetime.datetime(2020, 1, 1)
    #end_date = datetime.datetime(2021, 6, 1)
    #table = table.loc[(table.index > start_date) & (table.index <= end_date)]
    #all_user_sar = pd.read_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots/sar_ukclass_th-10_-10.csv')
    all_user_sar = pd.read_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots3/sar_ukclass_th-1001_-1001.csv')
    all_user_sar = all_user_sar[['Month', 'A']].rename(columns={'A':'Total User'})
    
    #table = table.head(26) #####mnth
    
    #table.to_csv('/disks/sda/adhiman/SAR-z/images/user_rawtweets/fsar_scores_'+str(thresh['author_threshold'])+"_"+str(thresh['family_threshold'])+"_"+'.csv')
    #table = table.merge(all_user_sar, on="Month")  ##mnth
    #table['User selected %'] = table['A']*100/table['Total User']  ##mnth
    
    if not os.path.exists(f'/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/class_sample/run_{r}'):
        os.makedirs(f'/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/class_sample/run_{r}')
        
    #table.to_csv(f"/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/class_sample/run_{r}/"+str(v)+"_sar_ukclass_th"+str(ath)+"_"+str(fth)+'.csv', index=None)
    table.to_csv(f'/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/shifting_classes/top_tweets/14_p05_all_twts_clean_newboost.csv', index=None)
    #remove
    #table.to_csv(f'/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/overlap/14day/psar/p005_{v}sar'+str(ath)+"_"+str(fth)+'.csv', index=None)
    #table.to_csv(f'/disks/sdb/adhiman/SAR-z/monthly_classified_user_days/weekly_analysis/14_p05_all_twts_clean_newboost.csv', index=None)
    
    fig, ax = plt.subplots(figsize=(12,7))
    
    #table[["NAIVE_SAR", 'SAR_adj_5']].rename({"NAIVE_SAR":"pSAR"},axis=1).plot(marker='o', ax=ax, color=['green', 'magenta', 'red'], linewidth=2) #####update
    
    #plt.hist(all_probs, bins=30)
    # set x-axis label
    '''ax.set_xlabel("Months",fontsize=18)'''
    # set y-axis label
    '''ax.set_ylabel("Predicted SAR",fontsize=18)'''
    '''ax.legend([ 'hSAR', 'PHE_SAR'], fontsize =24, loc=2)'''
    '''ax.set_ylim([0,.6])'''
    #ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    #table[["A"]].plot(marker='o', ax=ax2, color= 'blue')
    #ax2.set_ylim([0,15])
    #ax2.set_ylabel("Users",color="blue",fontsize=14)
    #ax2.legend(['Total Users'],loc=6, bbox_to_anchor=(0, 0.73)) #

    #table[["NAIVE_SAR","Elad_SAR","new_SAR", "PHE_SAR"]].rename({"NAIVE_SAR":"Baseline", "Elad_SAR": "Regression", "new_SAR": "Probability"},axis=1).plot(marker='o', ax=ax)
    #ax.set_ylabel("SAR estimate")
    #ax.fill_between(table.index, table.gold_min, table.gold_max, color='green', lw=2, alpha=0.1)
    '''plt.xticks(range(0,len(table.Month)), table.Month)'''
    #plt.xticks(range(0,len(table.index)), table.index)
    ax.legend(fontsize =18)
    #plt.xticks(rotation=90)
    '''plt.setp( ax.xaxis.get_majorticklabels(), rotation=70, fontsize =18 )'''
    '''plt.setp( ax.yaxis.get_majorticklabels(), fontsize =18 )'''
    plt.grid()
    #print ('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
    #print (path)
    #plt.savefig("/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/validation/prob_distr_all.pdf", bbox_inches='tight')
    ##removee
    #plt.savefig(f"/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/overlap/14day/psar/p005_{v}sar"+str(ath)+"_"+str(fth)+".pdf", bbox_inches='tight')
    #plt.savefig(f"/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/class_sample/run_{r}/"+str(v)+"_sar_ukclass_th"+str(ath)+"_"+str(fth)+".pdf", bbox_inches='tight')
    plt.close()
    return table
    
def plot_data2(tab, path):  
    tables = tab.copy()
    pd.concat([tables[key]["SAR_sum"].rename(f"SAR_{key}") for key in tables],axis=1).plot(color=[(1/(x*x/100+1), 1/(x*x/100+1), 1/(x*x/100+1)) for x in range(len(tables))], legend=False)
    plt.xlabel("")
    plt.ylabel("SAR estimate")
    plt.xticks(rotation='vertical')
    plt.savefig(f"{path}proba_convergence_tilloct"+".pdf", bbox_inches='tight')

    
def get_users():
    users_to_use = glob.glob('/disks/sda/adhiman/SAR-z/raw_tweets/users_to_download_all/users_to_download_after_classification/*.pkl')
    #all_users = read_pickle1(users_to_use[0])
    all_users = []
    for f in users_to_use:
        df = read_pickle1(f)
        all_users+=df.USER_ID.unique().tolist()
        #all_users = pd.concat([all_users, read_pickle1(f)])
        #print (all_users.shape)
    all_users = list(set(all_users))
    print (len(all_users)) 
    #users = all_users['USER_ID'].unique()
    ex_users = glob.glob('/disks/sda/adhiman/SAR-z/all_user_timelines/*.pkl')
    existing=[e.split('/')[-1].replace('.pkl','') for e in ex_users]
    #print (existing[:5])
    final_users = list(set(all_users).intersection(set(existing)))
    print (len(final_users))
    #print (final_users[:5])
    return (final_users)    

def get_users2(): #new
    users_to_use = glob.glob('/disks/sda/adhiman/SAR-z/ground_truth/UKlocations/loc_labled/*.pkl')
    #all_users = read_pickle1(users_to_use[0])
    all_users = []
    for f in users_to_use:
        df = read_pickle1(f)
        df  =df[df['UK']==True]
        all_users+=df.id.unique().tolist()
        #all_users = pd.concat([all_users, read_pickle1(f)])
        #print (all_users.shape)
    all_users = list(set(all_users))
    print (len(all_users)) 
    #users = all_users['USER_ID'].unique()
    #ex_users = glob.glob('/disks/sda/adhiman/SAR-z/all_user_timelines/*.pkl')
    ex_users = glob.glob('/disks/sdb/adhiman/uk_bow_vectors/*.pkl')
    existing=[e.split('/')[-1].replace('.pkl','') for e in ex_users]
    #print (existing[:5])
    final_users = list(set(all_users).intersection(set(existing)))
    print (len(final_users))
    #print (final_users[:5])
    return (final_users) 


def check_overlap(ukusers):
    fin_u =[]
    count =0
    user_name = open('/disks/sda/adhiman/SAR-z/ground_truth/userlist/overlapping2.txt', 'w')
    for u in ukusers:
        df = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors/{u}/api.pkl')
        t_u = df['TWEET_ID'].tolist()
        if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{u}/api.pkl'):
            count =count+1
            print (count)
            df_new = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{u}/api.pkl')
            t_u = t_u + df_new['TWEET_ID'].tolist()
        if len(set(t_u).intersection(set(raw_tweets)))!=0:
            #print ("adding")
            fin_u.append(u)
            user_name.write(str(u)+'\n')
            #print (len(final_users))
    user_name.close()
    return fin_u
            



def get_users3(sample, x): #new
    if sample==False:
        users_to_use = glob.glob('/disks/sda/adhiman/SAR-z/ground_truth/UKlocations/loc_labled/*.pkl')
        #all_users = read_pickle1(users_to_use[0])
        all_users = []
        for f in users_to_use:
            df = read_pickle1(f)
            df  =df[df['UK']==True]
            all_users+=df.id.unique().tolist()
            #all_users = pd.concat([all_users, read_pickle1(f)])
            #print (all_users.shape)
        all_users = list(set(all_users))
        print ("one")
        print (len(all_users)) 
        #users = all_users['USER_ID'].unique()
        ex_users = glob.glob('/disks/sda/adhiman/SAR-z/all_user_timelines/*.pkl')
        existing=[e.split('/')[-1].replace('.pkl','') for e in ex_users]
        #print (existing[:5])
        final_users = list(set(all_users).intersection(set(existing)))
        print ("two")
        print (len(final_users))
        
        #final_users1 = check_overlap(final_users)
        print ("overlapping users")
        print (len(final_users))
        #print (final_users[:5])
        return (final_users)
    
    if sample==True:
        print ("RUNNNING SAMPLING+++++++++++++++++++++++++++++++++++++++++++++++++++")
        users_to_use = glob.glob('/disks/sda/adhiman/SAR-z/ground_truth/UKlocations/loc_labled/*.pkl')
        #all_users = read_pickle1(users_to_use[0])
        all_users = []
        ex_users = glob.glob('/disks/sda/adhiman/SAR-z/all_user_timelines/*.pkl')
        existing=[e.split('/')[-1].replace('.pkl','') for e in ex_users]
        
        for f in users_to_use:
            df = read_pickle1(f)
            idx = f.split('/')[-1].replace('.pkl', '')
            df  =df[df['UK']==True]
            mnth_usr = df.id.unique().tolist()
            mnth_usr = list(set(mnth_usr).intersection(set(existing)))
            #print ("MONTH USER")
            #print (len(mnth_usr))
            try:
                all_users+= random.sample(mnth_usr, x)
            except:
                all_users+= mnth_usr
            #print ("ALL USER")
            #print (len(all_users))
        all_users = list(set(all_users))
        #print ("final length")
        #print (len(all_users))
        print ("using number of users", x)
        return (all_users)
    
def get_users4(sample, x):
    if sample ==False:
        user_ids = []
        #print (dfpr.info())
        #with open('/disks/sda/adhiman/SAR-z/ground_truth/userlist/overlapping.txt', 'r') as alus:
        #with open('/disks/sda/adhiman/SAR-z/ground_truth/userlist/overlapping_pr005.txt', 'r') as alus: 
        with open('/disks/sda/adhiman/SAR-z/ground_truth/userlist/overlapping_pr05.txt', 'r') as alus: 
        #with open('/disks/sda/adhiman/SAR-z/ground_truth/userlist/overlapping_pr0.txt', 'r') as alus:
        #with open('/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/validation/test_user.txt', 'r') as alus:
            for line in alus:
                uid = line.strip()
                user_ids.append(uid)
                
        user_ids = list(set(user_ids))
        print(len(user_ids))
        return user_ids
    if sample==True:
        user_ids = []
        users_to_use = glob.glob('/disks/sda/adhiman/SAR-z/ground_truth/UKlocations/loc_labled/*.pkl')
        
        with open('/disks/sda/adhiman/SAR-z/ground_truth/userlist/overlapping.txt', 'r') as alus:
            for line in alus:
                user_ids.append(line.strip())
        print(len(user_ids))
        all_users = []
        for f in users_to_use:
            df = read_pickle1(f)
            idx = f.split('/')[-1].replace('.pkl', '')
            df  =df[df['UK']==True]
            mnth_usr = df.id.unique().tolist()
            mnth_usr = list(set(mnth_usr).intersection(set(user_ids)))
            #print ("MONTH USER")
            #print (len(mnth_usr))
            try:
                #random.seed(27)
                all_users+= random.sample(mnth_usr, x)
            except:
                all_users+= mnth_usr
            #print ("ALL USER")
            #print (len(all_users))
        all_users = list(set(all_users))
        #print ("final length")
        #print (len(all_users))
        print ("using number of users", x)
        return (all_users)

    
    
def get_sample_users(x):
    all_users =[]
    all_usr_cls = defaultdict(list)
    for i in list(range(1,27,1)):
        mnth_user = []
        usr_cls = {}
        for mnth in glob.glob(f'/disks/sdb/adhiman/SAR-z/monthly_classified_user/{i}_*.txt'):
            clss= mnth.split('/')[-1].split('_')[1].replace('.txt', '')
            with open(mnth, 'r') as fp:
                for line in fp:
                    y = line[:-1]
                    mnth_user.append(y)
                    usr_cls[y]= clss
            #print (len(mnth_user))
            #random.seed(27)
            #print (mnth)
        #print ("keys")
        #print (len(usr_cls.keys()))
        
        if len(mnth_user)>=x:
            smpleuser = random.sample(mnth_user, x)
            
        else:
            print ("taking all")
            smpleuser = mnth_user
        
        #print ("sampled users")
        #print (len(smpleuser))
        
        #el2remv = set(all_users).intersection(set(smpleuser))
        #smpleuser = list(set(smpleuser) - set(el2remv))
        
        for u in smpleuser:
            all_usr_cls[u].append(usr_cls[u])
        
        #print ("filterd sampled users")
        #print (len(smpleuser))
        
        #print ("filterd sampled keys")
        #print (len(all_usr_cls.keys()))
        
        all_users += smpleuser
        
        #print ("filterd all users")
        #print (len(all_users))
        
    all_users = list(set(all_users))
    #print ('filtered keys')
    #print (len(all_usr_cls.keys()))
    #print ('filtered users')
    print (len(all_users)) 
    #all_users = list(set(all_users))
    return all_users, all_usr_cls
        
        
def basic_run( inpath, outpath,auth, fmth, v):
    print ("===================================================================================================================================================================")
    
    n_p = os.cpu_count()-1

    print ("utilizing number of cpus", n_p)
    
    user_ids= []
    #with open('/disks/sda/adhiman/SAR-z/users_tilloct2.txt', 'r') as alus:
    '''if os.path.exists(str(users)):
        print ('user_file input')
        with open(users, 'r') as alus:
            for line in alus:
                user_ids.append(line.strip())
        alus.close()

        user_ids = [str(user_id) for user_id in user_ids]
        print (user_ids[:5])

        for u in user_ids:
            if (u=='users') or (u=='users_to_download'):
                user_ids.remove(u)
                print ('here')
    else:
        user_ids = users'''
    
    '''    user_ids = glob.glob(inpath+'*')
    #print (user_ids[:5])
    user_ids=[u.split('/')[-1] for u in user_ids]'''
    #user_ids = get_users3(False, v)
    #user_ids = get_users4(True, v)
    user_ids = get_users4(False, v) #####change
    
    #user_ids = ['50989434']
    #user_ids = user_ids[:1000] #########update----------------------------------------------
    #user_ids = get_users3(True, v)
    
    #user_ids, user_clss = get_sample_users(v)
    
    '''ex_users = glob.glob('/disks/sda/adhiman/SAR-z/all_user_timeline_vectors/*')
    user_ids=[e.split('/')[-1].replace('.pkl','') for e in ex_users]
    print (ex_users[:4])
    print (user_ids[:4])'''
    
    
    print (len(user_ids))
    
    for u in user_ids:
        if (u=='users') or (u=='users_to_download'):
            user_ids.remove(u)
            print ('here')
    print (user_ids[:5])
    #user_ids = [ '33192983']#,'2712475177','2964957071','1080234963210067968'
    
    
    #path = '/disks/sda/adhiman/SAR-z/user_data_tilloct2/'
    path = inpath
    print (path)
    #for u in [str(1006091208106610688)]:
        #author_fam_positive(u, path)
    
    #out_path = '/disks/sda/adhiman/SAR-z/user_label_tilloct2/'
    out_path = outpath
    print (out_path)
    pool = Pool(n_p)
    
    print (list(zip(user_ids, [path]*len(user_ids), [out_path]*len(user_ids), [auth]*len(user_ids), [fmth]*len(user_ids)))[:2])
    
    classes = pool.starmap(author_fam_positive, zip(user_ids, [path]*len(user_ids), [out_path]*len(user_ids), [auth]*len(user_ids), [fmth]*len(user_ids)))
    pool.close()
    
    #print (classes[:6])
    
    summ, elad_w, si, all_probs = calSAR(classes)
    
    #print (summ)
    #print (summ.apply(pd.Series.value_counts).T)
    #summ.index = summ.index.astype('str')
    #print (summ.info())
    #summ.to_csv("../../../disks/sda/adhiman/SAR-z/temp_out_aux/az_classes.csv")
    
    gold_std = gold()
    #print (gold_std)
    
    #summ =pd.read_csv("../../../../disks/sda/adhiman/SAR-z/SAR_classes.csv")
    #summ = summ.set_index('Unnamed: 0')
    #print (summ)
    
    elad_res = elad_sar2(summ, elad_w)
    #print (elad_res)
    pred_res = pred_sar(summ, elad_res, gold_std,si, len(user_ids))
    #print (pred_res)
    
     ####################################
    '''tables ={}
    for i in range(1000, 5500, 500):
        if i < summ.shape[0]: 
            table = pred_sar(summ.sample(i), elad_res, gold_std)
        else:
            table = pred_sar(summ, elad_res, gold_std)
        tables[i] = table'''
    ############################################
    gr = prep_ground()
    #print (gr)
    #print (gr.info())
    #print (pred_res.info())
    gr['Month'] =gr['Month'].astype('str')
    pred_res['Month'] =pred_res['Month'].astype('str')
    #print (pred_res)
    #pred_res = pred_res.merge(gr[['Month','PHE_SAR']], left_index = True, right_on='Month', how ='left')
    #pred_res = pred_res.merge(gr[['Month','PHE_SAR', 'alpha_sar', 'delta_sar']], on='Month', how ='left')
    ##pred_res = pred_res.merge(gr[['Month','SAR_adj_5', 'SAR_5']], on='Month', how ='left') ##mnth
    
    ##pred_res =pred_res.set_index('Month')  ##mnth
    #pred_res['PHE_SAR'] = pred_res['PHE_SAR'].shift(periods=-1)
    #print (pred_res)
    #pred_res.to_csv(out_path+'../images/fsar'+'.csv')
    return pred_res, all_probs
    #
    #plot_data(pred_res, path, r)
    
    #plot_data2(tables, path)
    
    #print (summ.shape)
    #print (summ.info())
    #summ = summ.sort_index(ascending=False)
    #summ.to_csv("../../../../disks/sda/adhiman/SAR-z/SAR_classes.csv")
    

def cal_avg(all_runs):
    df_run = all_runs.copy()
    #print (df_run)
    
    df_run['avg'] = all_runs.mean(axis=1)
    df_run['std'] = all_runs.std(axis=1)
    df_run['min']= all_runs.min(axis=1)
    df_run['max']= all_runs.max(axis=1)
    #print (df_run)
    gr = prep_ground()
    #print (gr)
    df_run = df_run.merge(gr[['SAR_adj_5', 'SAR_5','Month', 'Users']], right_on ='Month', left_on=df_run.index, how ='left')
    df_run.to_csv('/disks/sda/adhiman/SAR-z/samples_output_tilloct2/samples_sar.csv')
    #print (df_run)
    
    
    
def sample_run2(user_dir, inpath, outpath, runs):
    files = glob.glob(user_dir+'/*.pkl')
    #print (files)
    print (outpath)
    users = []
    all_runs = pd.DataFrame()
    for r in range(int(runs)):
        users = []
        for file in files:
            df_temp = read_pickle1(file)
            if df_temp.shape[0]>=300:
                temp_usrs= df_temp.sample(n=300)[0].tolist()  
            else:
                temp_usrs = df_temp[0].tolist()
            print (len(temp_usrs))
            users = users+temp_usrs
        users=list(set(users))
        print (len(users))
        out =outpath
        out = outpath+'sample_'+str(r)+'/'
        
        if not os.path.exists(out):
            os.mkdir(out)
        res = basic_run(users, inpath, out, r)
        
        all_runs[r] = res['new_SAR']
        #print (res['new_SAR'])
   
    cal_avg(all_runs)
    
    
    
    
def sample_run(user_dir, inpath, outpath, runs):
    files = glob.glob(user_dir+'/*.pkl')
    #print (files)
    print (outpath)
    users = []
    all_runs = pd.DataFrame()
    for r in range(int(runs)):
        users = []
        for file in files:
            df_temp = read_pickle1(file)
            if df_temp.shape[0]>=300:
                temp_usrs= df_temp.sample(n=300)[0].tolist()  
            else:
                temp_usrs = df_temp[0].tolist()
            print (len(temp_usrs))
            users = users+temp_usrs
        users=list(set(users))
        print (len(users))
        out =outpath
        out = outpath+'sample_'+str(r)+'/'
        
        if not os.path.exists(out):
            os.mkdir(out)
        res = basic_run(users, inpath, out, r)
        
        all_runs[r] = res['new_SAR']
        #print (res['new_SAR'])
   
    cal_avg(all_runs)
    

    
    
def main():
    
    args = sys.argv[1:]
    print (args)
    if len(args)>5:
        print ('invalid parameters received')
        return
    elif args[0] == str(1):
        print ('here')
        th_uslctd = []
        #for x,y in list((itertools.permutations(np.arange(start= -0.01, stop=-4, step=-0.2), 2))):
        #for x in list(np.arange(start= -0.11, stop=-4, step=-0.2)):
        auth_perf = pd.read_csv('/disks/sdb/adhiman/SAR-z/SARplots2/ct_plots/acc_per_thrsh/auth.csv')
        fam_perf = pd.read_csv('/disks/sdb/adhiman/SAR-z/SARplots2/ct_plots/acc_per_thrsh/fam.csv')
        
        #for ths in tqdm(list(itertools.product(auth_perf['Threshold'].tolist(),fam_perf['Threshold'].tolist()))):
        #for ths in [(-1.81, -0.16), (-1.96, -0.16),(-1.66, -0.16),(-1.81, -0.31),(-1.96, -0.31),(-1.66, -0.31),(-1.81, -0.46),(-1.96, -0.46),(-1.66, -0.46)]:
        #for ths in tqdm(list(itertools.product(auth_perf['Threshold'].tolist(),fam_perf['Threshold'].tolist()))):\
        #for x,y in list((itertools.permutations(np.arange(start= 0.6, stop=2.7, step=0.1), 2))):
        #f = np.arange(start= 0.1492, stop=2.1492, step=0.1)
        #a = np.arange(start= -0.5584, stop=1.4416, step=0.1)
        #f = np.arange(start= 0.028, stop=1.97, step=0.1)
        #a = np.arange(start= -0.73, stop=1.26, step=0.1)
        #for x,y in list(itertools.product(a,f)):
        for run in [1]: #list(range(0,50,1))
            for v in [14]:
                x =-0.63
                y = 0.028
                #y = 2
                #v=0
                print ("=====Thresholds======")
                #x =ths[0]
                #y =ths[1]
                print (x)
                #x= -0.22
                #y= -0.56
                print (y)
                if not os.path.exists(args[2]):
                    os.makedirs(args[2])
                else:
                    shutil.rmtree(args[2])           # Removes all the subdirectories!
                    os.makedirs(args[2])
                fsar, all_probs = basic_run(args[1], args[2],x,y,v)
                data_out = plot_data(fsar, all_probs, args[1], x,y,v, r=run)
                
                '''a_p = list(auth_perf[auth_perf['Threshold']==x].squeeze('rows').values)
                f_p = list(fam_perf[fam_perf['Threshold']==y].squeeze('rows').values)
                temp_usr = a_p+f_p+data_out['User selected %'].tolist()+data_out['A'].tolist()
                th_uslctd.append(temp_usr)

        
        df_perf_useld = pd.DataFrame(th_uslctd, columns = ['A_Thrsh',  'A_Precision', 'A_Recall', 'A_F1', 'A_Accuracy', 'A_Labeled Negative', 'A_Labeled Positive','F_Thrsh', 'F_Precision', 'F_Recall', 'F_F1', 'F_Accuracy', 'F_Labeled Negative', 'F_Labeled Positive', 'Jan-20 %','Feb-20 %','Mar-20 %','Apr-20 %','May-20 %','Jun-20 %','Jul-20 %','Aug-20 %','Sep-20 %','Oct-20 %','Nov-20 %','Dec-20 %','Jan-21 %','Feb-21 %','Mar-21 %','Apr-21 %','May-21 %','Jun-21 %','Jul-21 %', 'Aug-21 %','Sep-21 %','Oct-21 %','Nov-21 %','Dec-21 %','Jan-22 %','Feb-22 %', 'Jan-20','Feb-20','Mar-20','Apr-20','May-20','Jun-20','Jul-20','Aug-20','Sep-20','Oct-20','Nov-20','Dec-20','Jan-21','Feb-21','Mar-21','Apr-21','May-21','Jun-21','Jul-21', 'Aug-21','Sep-21','Oct-21','Nov-21','Dec-21','Jan-22','Feb-22'])
        
        df_perf_useld.to_csv('/disks/sdb/adhiman/SAR-z/SARplots2/ct_plots/thsh_perf_usrslctd2.csv', index=False)
        '''
        '''x =-50
        fsar = basic_run(args[1], args[2],x,x, r=-1)
        plot_data(fsar,  args[1], x,x, r=-1)'''
    elif args[0] == str(2):
        sample_run(args[1], args[2], args[3], args[4])
    else:
        print ('invalid argument')
        return
    
    
    
    
if __name__ == '__main__':
    
    print ('give users; input folder and output folder')
    
    main()
    #get_users()
    
    
    
    
