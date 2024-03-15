#This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).

import pandas as pd
import glob
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


#label ='ABT_FAMILY'

label ='FAMILY_OR'

df_all = pd.DataFrame()


f_all = []
a_all = []
p_all = []
r_all = []


th =defaultdict(list)
for p in glob.glob('family_val/*.csv'):
    df_val = pd.read_csv(p)
    df_val['log']= np.log(df_val['Predictions'])
    df_val['norm'] = (df_val['Predictions']-df_val['Predictions'].min())/(df_val['Predictions'].max()-df_val['Predictions'].min())
    df_val['log_norm'] = (df_val['log']-df_val['log'].min())/(df_val['log'].max()-df_val['log'].min())
    f1_f = -1
    th_f=0
    acc_f=-1
    pre_f=-1
    rec_f=-1
    for n in np.arange(df_val['log'].min(),df_val['log'].max(), 0.01):
        pred_class = np.array(df_val['log'])>n
        f1 = f1_score(df_val[label], pred_class, average ='macro')
        acc =accuracy_score(df_val[label], pred_class)
        prc =precision_score(df_val[label], pred_class, average ='macro')
        rec =recall_score(df_val[label], pred_class, average ='macro')
        if f1>=f1_f:
            f1_f=f1                   
            th_f = n
            acc_f = acc
            pre_f = prc
            rec_f = rec
    train_id = p.split('/')[-1].split('_')[1]
    th[train_id].append(th_f)
    
    


final_th={}
for th_k in list(th.keys()):
    final_th[th_k] = np.mean(th[th_k])
    



f1_all = []
ac_all=[]
prc_all=[]
rc_all = []

for path in glob.glob('fam_test/*'):
    df = pd.read_csv(path)
    
    #idx = path.split('/')[-1].replace('.csv', '').replace('preds_', '')
    idx = path.split('/')[-1].replace('.csv', '')
    df_val =pd.read_csv(path.replace('abt_family_test2', 'abt_family_val2'))
    df_val['log']= np.log(df_val['Predictions'])
    
    train_id = path.split('/')[-1].split('_')[1]
    
    df['log']= np.log(df['Predictions'])
    df['norm'] = (df['Predictions']-df_val['Predictions'].min())/(df_val['Predictions'].max()-df_val['Predictions'].min())
    df['log_norm'] = (df['log']-df_val['log'].min())/(df_val['log'].max()-df_val['log'].min())
    
    #print (idx)
    #print (type(idx))
    #print (th[idx])
    
    #pred_class = df['log']>th[idx]
    pred_class = df['log']>final_th[train_id]
    #pred_class = df['Sig_Predictions']>th[idx]
    

    f1 = f1_score(df[label], pred_class, average ='macro')
    acc =accuracy_score(df[label], pred_class)
    prc =precision_score(df[label], pred_class, average ='macro')
    rec =recall_score(df[label], pred_class, average ='macro')
    f1_all.append(f1)
    ac_all.append(acc)
    prc_all.append(prc)
    rc_all.append(rec)

print (f1_all[:10])
print ("f1 \n"+str(np.mean(f1_all))+'\t'+str(np.std(f1_all)))
print ("acc \n"+str(np.mean(ac_all))+'\t'+str(np.std(ac_all)))
print ("prc \n"+str(np.mean(prc_all))+'\t'+str(np.std(prc_all)))
print ("rec \n"+str(np.mean(rc_all))+'\t'+str(np.std(rc_all)))


