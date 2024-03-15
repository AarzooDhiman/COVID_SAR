# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).

import pandas as pd
import glob
from sklearn import metrics
import numpy as np


thresh = pd.read_csv('fam_thresh.csv')
print (thresh.head())

auc_lst = []

for i in range(10):
    
    test_df = pd.DataFrame()
    for f in glob.glob('fam_test/*.csv'):
        df_temp = pd.read_csv(f)
        
        #print (df_temp.tail())
        idx = f.split('/')[-1].replace('.csv', '')
        test_idx = int(idx.split('_')[1])
        train_idx = int(idx.split('_')[2])
        val_idx = int(idx.split('_')[3])

        
        th = thresh[(thresh['Test'] == test_idx) & (thresh['Train'] == train_idx) & (thresh['Val'] == val_idx)]['Thresh'].values[0]
        #print (th)
        df_temp['PRED_Label'] =df_temp['Sig_Predictions']>th
        test_df = pd.concat([test_df, df_temp])
        #print (test_df.shape)
        #print (test_df.tail())
        
        
    #print (test_df.head()) 
    test_df = test_df.sample(frac=1)
    #print (test_df.head())

    #print (test_df.shape)
    test_df =test_df.drop_duplicates('TWEET_ID')
    #print (test_df.shape)
    
    test_df = test_df[:1180]

    fpr, tpr, thresholds = metrics.roc_curve(test_df['FAMILY_OR'], test_df['Predictions'], pos_label=True)
    auc = metrics.auc(fpr, tpr)
    print (auc)
    auc_lst.append(auc)

print (np.mean(auc_lst))
