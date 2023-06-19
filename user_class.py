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



def find_best_pair(auth_df, fam_df, days):
    
    auth_df1 = auth_df[['TWEET_ID', 'USER_ID', 'AUTHOR_SCORE' , 'MONTH_ID', 'TIMESTAMP', 'PROB_A']]
    auth_df1= auth_df1.rename(columns={'TWEET_ID':'AUTH_TWEET_ID', 'MONTH_ID':'AUTH_MONTH_ID', 'TIMESTAMP':'AUTH_TIMESTAMP' })
    fam_df1 = fam_df[['TWEET_ID', 'USER_ID', 'FAMILY_SCORE' , 'MONTH_ID', 'TIMESTAMP', 'PROB_F']]
    fam_df1= fam_df1.rename(columns={'TWEET_ID':'FAM_TWEET_ID', 'MONTH_ID':'FAM_MONTH_ID', 'TIMESTAMP':'FAM_TIMESTAMP' })

    auth_df1 = auth_df1.sort_values(['AUTHOR_SCORE'], ascending=False) ##finding maximum
    fam_df1 = fam_df1.sort_values(['FAMILY_SCORE'], ascending=False) ##finding maximum
    
    df_all = auth_df1.merge(fam_df1, how='outer')
    
    df_all['delta'] = (df_all['FAM_TIMESTAMP'] - df_all['AUTH_TIMESTAMP']).dt.days
    df_all['PROB'] = df_all['PROB_A']*df_all['PROB_F']
    df_all = df_all.sort_values(['PROB'], ascending=False).reset_index()

    df_all1 = df_all[(df_all['delta']>=(0-days)) & (df_all['delta']<=days) & (df_all['delta']!=0)]

    
    if df_all1.shape[0]>0:
        final_delta =df_all1['delta'].values[0]
        neg_values = df_all1[df_all1['delta']<0].shape[0]
        pos_values = df_all1[df_all1['delta']>0].shape[0]
        
        final_auth_mnth = df_all1['AUTH_MONTH_ID'].values[0]
        final_fam_mnth = df_all1['FAM_MONTH_ID'].values[0]

        return (final_delta, final_auth_mnth, final_fam_mnth, df_all1['PROB_A'].values[0] , df_all1['PROB_F'].values[0] )
    
    elif df_all1.shape[0]==0:
        if df_all.shape[0]>0:
            return (df_all['delta'].values[0], df_all['AUTH_MONTH_ID'].values[0], df_all['FAM_MONTH_ID'].values[0], df_all['PROB_A'].values[0] , df_all['PROB_F'].values[0] )
        else:
            return "nothing"
        
def classes(author_positive_df, family_positive_df, fam_cnf, auth_cnf, path, user):
    
    author_positive = author_positive_df.copy()
    family_positive = family_positive_df.copy()

    classification = {}
    serial_interval = {}
    
    #print (author_positive)
    #print (family_positive)
    
    #start_of_timeline = datetime.datetime(2020, 1, 1, tzinfo=pytz.UTC) ##weekly values
    
    if author_positive.shape[0]>0:

        
        author_positive.TIMESTAMP = pd.to_datetime(author_positive.TIMESTAMP, utc=True)

        author_positive["MONTH_ID"] = author_positive.TIMESTAMP.dt.month + (author_positive.TIMESTAMP.dt.year-2020)*12 ##mnth
        
        '''author_positive['delta'] = author_positive.TIMESTAMP - start_of_timeline ##week
        author_positive['week'] = (author_positive['delta'].dt.days // 7) + 1 ##week
        author_positive['MONTH_ID'] = author_positive['week']''' ##week

        
        author_positive= author_positive.rename(columns={'log':'AUTHOR_SCORE'})
        
        author_positive = author_positive.sort_values("AUTHOR_SCORE", ascending=False).reset_index() ##finding maximum
        
        author_positive['PROB_A'] =  0.5 + 0.5*((author_positive['AUTHOR_SCORE']- tau_a)/(max_a - tau_a))

        
    if family_positive.shape[0]>0:
        #print (family_positive)
        family_positive.TIMESTAMP = pd.to_datetime(family_positive.TIMESTAMP, utc=True)
        
        #print ("fam shape")
        #print (family_positive.shape[0])
        
        family_positive["MONTH_ID"] = family_positive.TIMESTAMP.dt.month + (family_positive.TIMESTAMP.dt.year-2020)*12 #mnth
        
        '''family_positive['delta'] = family_positive.TIMESTAMP - start_of_timeline ##week
        family_positive['week'] = (family_positive['delta'].dt.days // 7) + 1
        family_positive['MONTH_ID'] = family_positive['week']'''
        
        
        
        family_positive= family_positive.rename(columns={'log':'FAMILY_SCORE'})
        
       
        family_positive = family_positive.sort_values("FAMILY_SCORE", ascending=False).reset_index()
        
        family_positive['PROB_F'] =  0.5 + 0.5*((family_positive['FAMILY_SCORE']- tau_f)/(max_f - tau_f))

    
    wd={}

    class_p=0
    
    class_p={}
    
    month_ids = {'auth_month':[], 'fam_month':[]}    
    
    if author_positive.shape[0]>0 and family_positive.shape[0]==0:

        auth_month_id = int(author_positive.iloc[0]['MONTH_ID'])

        classification[auth_month_id]=str(10)
        wd[auth_month_id] = auth_cnf
        p_val = author_positive.iloc[0]['PROB_A']
        month_ids['auth_month'] = author_positive['MONTH_ID'].tolist()
        class_p[auth_month_id] = p_val

    if author_positive.shape[0]==0 and family_positive.shape[0]>0:
        fam_month_id = int(family_positive.iloc[0]['MONTH_ID'])

        classification[fam_month_id]=str(20)
        wd[fam_month_id] = fam_cnf
        p_val = family_positive.iloc[0]['PROB_F']
        month_ids['fam_month'] = family_positive['MONTH_ID'].tolist()
        class_p[fam_month_id] = p_val

    if author_positive.shape[0]>0 and family_positive.shape[0]>0:

        days = 14
        delta, auth_month_id, fam_month_id, p_a, p_f = find_best_pair(author_positive, family_positive, days)
        
      
        month_ids['auth_month'] = author_positive['MONTH_ID'].tolist()
        month_ids['fam_month'] = family_positive['MONTH_ID'].tolist()


   

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

        elif delta!=0:                                          
            classification[auth_month_id] = str(10)
            wd[auth_month_id] = auth_cnf
            class_p[auth_month_id] = p_a
            
            classification[fam_month_id] = str(20)
            wd[fam_month_id] = fam_cnf
            class_p[fam_month_id] = p_a

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
    if os.path.exists(f'uk_user_positives/{user}/author_positive.csv'):
        author_positive = pd.read_csv(f'uk_user_positives/{user}/author_positive.csv')
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

    if os.path.exists(f'uk_user_positives/{user}/family_positive.csv'):
        family_positive= pd.read_csv(f'/uk_user_positives/{user}/family_positive.csv')
        
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
    except:
        return
   
    
    if os.path.exists(f'{path}{user}/author_score.pkl'):
        author_tweets = read_pickle1(f'{path}{user}/author_score.pkl')
        
        author_tweets['log'] = np.log(author_tweets['T_PRED'].astype('float32'))
        author_tweets= author_tweets.fillna(-100)
        author_tweets = author_tweets[['log', 'PRED', 'T_PRED']]
    else:
        author_tweets = pd.DataFrame()
    
    if os.path.exists(f'{path}{user}/fam_score.pkl'):
        family_tweets = read_pickle1(f'{path}{user}/fam_score.pkl')
        
        family_tweets['log'] = np.log(family_tweets['T_PRED'].astype('float32'))
        family_tweets= family_tweets.fillna(-100)
        family_tweets = family_tweets[['log', 'PRED', 'T_PRED']]
        
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
    

    
    author_positive =pd.DataFrame()
    family_positive =pd.DataFrame()
    
    #print ("here")

    
    if author_tweets.shape[0]>0:

        
        auth_max = author_tweets[author_tweets['log'] >= thresh['author_threshold']] 
        
  
        auth_max = auth_max.reset_index()
        
        if auth_max.shape[0]>0:
            if not os.path.exists(out_path+str(user)):
                os.mkdir(out_path+str(user))

    if family_tweets.shape[0]>0:
        family_max = family_tweets[family_tweets['log'] >= thresh['family_threshold']]
        
        family_max = family_max.reset_index()
        if family_max.shape[0] > 0:
            #print ('here')
            if not os.path.exists(out_path+str(user)):
                os.mkdir(out_path+str(user))
   
    auth_conf = 0
    fam_conf = 0
    exist=True

    if author_tweets.shape[0]>0:
        if auth_max.shape[0] > 0:
            
            auth_max['TWEET_ID'] = auth_max['TWEET_ID'].astype('str')


            author_positive = auth_max.merge(tweets, on='TWEET_ID',how='inner' )
            

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
                if not os.path.exists(f'uk_user_positives/{user}/'):
                    os.mkdir(f'uk_user_positives/{user}/')
                #print (author_positive)
                author_positive.to_csv(f'uk_user_positives/{user}/author_positive.csv')
            else:
                author_positive = pd.DataFrame()
  
            if author_positive.shape[0]>0:
                auth_conf = author_positive['PRED'].values[0]
            else:
                auth_conf = min_a
                
        else:
            exist = False
            
            

    if os.path.exists(f'{path}{user}/abtfam_score.pkl'):
        abt_fam = read_pickle1(f'{path}{user}/abtfam_score.pkl')
        if os.path.exists(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/abtfam_score.pkl'):
            abt_new_twts = read_pickle1(f'/disks/sdb/adhiman/SAR-z/uk_bow_vectors2/{user}/abtfam_score.pkl')
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
            
            family_positive = family_positive[family_positive['TWEET_ID'].isin(abt_fam_twts)]
            
            
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

                family_positive.to_csv(f'uk_user_positives/{user}/family_positive.csv')

            else:
                family_positive = pd.DataFrame()

            
            if family_positive.shape[0]>0:
                fam_conf = family_positive['PRED'].values[0]
          
            else:
                fam_conf = min_f
                
        else:
            exist = False
    
    if len(path1)>0:
        class_assg, eld_w, serial_interval, class_p = classes(author_positive, family_positive, fam_conf, auth_conf,  path1, user)
        if len(class_assg)>0:
            return (user,eld_w, p_f, class_assg, serial_interval, class_p)

def calSAR(classes0):
    classes = classes0.copy()

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
           
    summary = pd.DataFrame.from_dict(final_classes, orient="index")
    rate = pd.DataFrame.from_dict(pr_f, orient="index", columns = ['RATE'])
    si_df = pd.DataFrame.from_dict(si, orient="index")
    si_df.to_csv('top_tweets/user_si_p0.csv')
    
    si_df = si_df.apply(lambda x:x.mean(), axis =0)
    si_df = si_df.to_frame().rename(columns= {0:'si'})

    
    w_df = pd.DataFrame.from_dict(w, orient="index")
    w_df = w_df.reset_index()
    w_df = w_df.rename(columns ={'index':'user'})

    summary= summary.sort_index(axis=1)
    
    prob_sum = pd.DataFrame( columns=['10', '20', '12', '21'], index = list(range(29)))
    
    prob_sum = prob_sum.replace(np.nan, 0.0)
    
    
    for pair, clss in final_classes.items():
        for m, c  in clss.items():
            prob_sum.iloc[int(m)-1][c] =prob_sum.loc[int(m)-1][c] + all_probs[pair][m]
            
                
    prob_sum['fSAR'] = (prob_sum['12']+prob_sum['21'])/(prob_sum['12']+prob_sum['21']+prob_sum['10']+prob_sum['20'])         
    
    summary = summary.merge(rate,left_index=True, right_index=True)
   
    return summary, w_df, si_df, all_probs


    
def elad_sar2(summ, w_df):
    summary = summ.copy()
   
    elad_SARs = {}
    
    psar = {}
    c_val =list(summary.columns.values)
    c_val.remove('RATE')
 
    p_x = summary[summary['RATE']!=0.0].shape[0]/77016
    
  
    
    for month_ID in c_val:
        #print ("+++++++++++++++++++++++++++++++++++++++++")
        m_u =pd.DataFrame(summary[month_ID]).rename(columns={month_ID:'T'}).dropna(subset=['T'])
        w_u = w_df[['user',month_ID]].rename(columns={month_ID:'conf'})
       
        summary2 = summary.merge(w_u, left_on=summary.index, right_on='user')
        
        
        mnth_sum = summary2.merge(m_u, left_on='user', right_on =m_u.index )

        
        T_R= mnth_sum['T'].replace('12', 1.0).replace('21', 1.0).replace('10', 0.0).replace('20',0.0)
        
        T_R = T_R.fillna(0.0)
        T_R = T_R.astype('float64')
       
        T_R = T_R.to_numpy()
       
        P_R = np.array([p_x]*mnth_sum.shape[0])
        
        #P_R = np.array([0.87065]*mnth_sum.shape[0])
        W = np.diag(mnth_sum.conf)

        p_left = np.matmul(np.matmul(P_R, W), P_R.T)
        
        p_right = np.matmul(np.matmul(P_R,W), T_R.T)
        
        psar[month_ID] = p_right/p_left
        
    Elad_SAR_df = pd.DataFrame.from_dict(psar, orient="index").rename({0: "new_Elad_SAR"}, axis=1)
    
    return Elad_SAR_df

def pred_sar(summ,elad, gold_df, si, total_users):
    
    summary = summ.copy()
    
    p_x = summary[summary['RATE']!=0.0].shape[0]/77016

   
    class_summ = summary.drop(columns=['RATE']).apply(pd.Series.value_counts).T
    
   
    print ('==================================')

    p= {}
    p_2 ={}
    p_df = pd.DataFrame()
    for x in summary.drop(columns= ['RATE']).columns.values:
       
        y = summary[[x,'RATE']].replace('10', np.nan).replace('20', np.nan).replace('11020', np.nan).replace('21020', np.nan) ###update
        
        y = y.dropna()
       
        c = summary[[x]].count().values[0]
        p_u = y['RATE'].sum()/y.shape[0]
        p[x] = p_u
       
        valid_users = summary[[x]].dropna().index.tolist()
       
        y2 = summary[summary.index.isin(valid_users)][[x,'RATE']]
        
        p_u2 = y2[y2['RATE']>0.0].shape[0]/summary.shape[0]  
       
        p_2[x] = p_u2

    p_df = pd.DataFrame.from_dict(p, orient='index', columns = ['P_U'])
    
    p_df2 = pd.DataFrame.from_dict(p_2, orient='index', columns = ['P_U2'])
   
    class_summ =class_summ.merge(si, left_index =True, right_index=True )
   
    
    class_summ =class_summ.merge(p_df2, left_index =True, right_index=True )
    class_summ = class_summ.merge(p_df, left_index = True, right_index = True)
    
    table =class_summ.copy()
    table =table.fillna(0)
    
    for cl in ['10','20','12','21']: 
        if cl not in table.columns.values:
            table[cl]=0
    table["A2"] = table['21']+table['20']
    table["A1"] = table['12']+table['10']
    table["A"] = table['12']+table['10']+table['21']+table['20']
   
    table["S1"] = (table['12']+table['10'])/(table['12']+table['10']+table['21']+table['20'])
    
    table["S2"] = (table['21']+table['20'])/(table['12']+table['10']+table['21']+table['20'])
   
    table['alpha2'] = (table['21'])/(table['21']+table['20'])
    table['alpha1'] = (table['12'])/(table['12']+table['10'])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(table['alpha1'].tolist(), table['alpha2'].tolist())
  
   
    table['p_alpha1'] = slope*table['alpha1']+intercept
    
    table['fSAR'] = table["S1"]*table['alpha1']+table["S2"]*table['alpha2']
    table['pSAR'] = table["S1"]*table['p_alpha1']+table["S2"]*table['alpha2']
   
    table["NAIVE_SAR"] = (table['12'] + table['21']) / table["A"]
    
    table = table.join(elad)
   
    table = table.join(gold_df)
   
    datetime_series = pd.Series(pd.date_range("2019-12-01", periods=table.shape[0]+1, freq="M")) #+int(table.index[0]) ###mnth    
    
    datetime_series = datetime_series.dt.strftime("%b-%y")[table.index[0]:] ###mnth

    
    #datetime_series = pd.Series(pd.date_range("2020-01-01", periods=table.shape[0]+1, freq="W")) ##week
    #datetime_series =datetime_series+datetime.timedelta(days=2) #week
    #datetime_series = datetime_series.dt.strftime("%d-%b-%y")[table.index[0]:] ###week
    #datetime_series.index += 1 
    
    
    table = table.merge(datetime_series.to_frame(), left_index=True, right_index=True) ##mnth
    #table['Week'] = datetime_series.dt.strftime("%d-%b-%y") ##week
    table = table.rename(columns={0:'Month'}) ##mnth
    
    #table.index = datetime_series
    print (table)
    
   
    return table
   
    
    
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
    
    table.to_csv(f'/SAR_scores.csv', index=None)
    return table
    

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
            
    
def get_users4():

    user_ids = []
    with open('/overlapping_pr0.txt', 'r') as alus: 
        for line in alus:
            uid = line.strip()
            user_ids.append(uid)
            
    user_ids = list(set(user_ids))
    print(len(user_ids))
    return user_ids
    
        
def basic_run( inpath, outpath,auth, fmth, v):
    print ("===================================================================================================================================================================")
    
    n_p = os.cpu_count()-1

    print ("utilizing number of cpus", n_p)
    
    user_ids= []
    
    user_ids = get_users4() 
    
    
    for u in user_ids:
        if (u=='users') or (u=='users_to_download'):
            user_ids.remove(u)
            print ('here')
    print (user_ids[:5])
    
    path = inpath
    print (path)
    
    out_path = outpath
    print (out_path)
    pool = Pool(n_p)
    
    print (list(zip(user_ids, [path]*len(user_ids), [out_path]*len(user_ids), [auth]*len(user_ids), [fmth]*len(user_ids)))[:2])
    
    classes = pool.starmap(author_fam_positive, zip(user_ids, [path]*len(user_ids), [out_path]*len(user_ids), [auth]*len(user_ids), [fmth]*len(user_ids)))
    pool.close()
    
    
    summ, elad_w, si, all_probs = calSAR(classes)
    
    
    gold_std = gold()
   
    
    elad_res = elad_sar2(summ, elad_w)
   
    pred_res = pred_sar(summ, elad_res, gold_std,si, len(user_ids))
    

    ############################################
    gr = prep_ground()
  
    gr['Month'] =gr['Month'].astype('str')
    pred_res['Month'] =pred_res['Month'].astype('str')
   
    return pred_res, all_probs

    
def main():
    
    args = sys.argv[1:]
    print (args)
    if len(args)>5:
        print ('invalid parameters received')
        return
    elif args[0] == str(1):
        print ('here')
        th_uslctd = []
        

        for run in [1]: #list(range(0,50,1))
            for v in [14]:
                x =-0.63
                y = 0.028
                print ("=====Thresholds======")

                if not os.path.exists(args[2]):
                    os.makedirs(args[2])
                else:
                    shutil.rmtree(args[2])           # Removes all the subdirectories!
                    os.makedirs(args[2])
                fsar, all_probs = basic_run(args[1], args[2],x,y,v)
                data_out = plot_data(fsar, all_probs, args[1], x,y,v, r=run)
                

    elif args[0] == str(2):
        sample_run(args[1], args[2], args[3], args[4])
    else:
        print ('invalid argument')
        return
    
if __name__ == '__main__':
    
    print ('give users; input folder and output folder')
    
    main()
    
    
    
    
