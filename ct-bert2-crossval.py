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
from sklearn.model_selection import StratifiedKFold

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
from tqdm import tqdm
from preprocess import clean_tweets
from sklearn.metrics import f1_score
import sys
from sklearn.utils.class_weight import compute_class_weight

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
import eda_supplement


print (torch.__version__)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

print ("====================================================================================================================")
#print ("using gpu: ", torch.cuda.get_device_name(1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_pickle1(pth):
    with open(pth, "rb") as fh:
        data = pickle.load(fh)
    return (data)


def datasplit(label, feat):
    
    #df = read_pickle1('/disks/sdb/adhiman/SAR-z/labelbox itr3/with consensus/data_for_validation/itr123_train.pkl')
    df = read_pickle1('/disks/sdb/adhiman/SAR-z/labelbox itr3/with consensus/data_for_validation/itr123.pkl')
    df['ABT_FAMILY'] = df['FAMILY'].astype(bool)
    df = clean_tweets(df,False, remove_duplicates=False)
    
    
    '''df1 = read_pickle1('/home/adhiman/SAR-z/to_label_data/code/final_labels/labeled_SET1.pkl')
    df2 = read_pickle1('/disks/sdb/adhiman/SAR-z/labelbox itr2/with_consensus/itr2.pkl')
    df2 = df2[list(df1.columns.values)]
    df2 = df2.replace('NO', 0)
    df2 = df2.replace('YES', 1)
    df = pd.concat([df1,df2])
    '''

    '''df = read_pickle1('/home/adhiman/SAR-z/labeled data/itr1_2.pkl')
    
    #print (df.shape)
    #print (df.columns.values)
    #df_test = pd.read_csv('/disks/sdb/adhiman/SAR-z/labelbox itr3/without consesus/matching/data_itr3_mtch.csv')
    df_test = pd.read_csv('/disks/sdb/adhiman/SAR-z/labelbox itr3/with consensus/itr3.csv')
    
    df_test = df_test.replace('NO', 0).replace('YES', 1)
    df_test=df_test[['TWEET_ID', 'USER_ID', 'TIMESTAMP' ,'TWEET_TEXT' ,'AUTHOR_COVID', 'AUTHOR_SYMPTOMS' ,'FAMILY', 'FAMILY_COVID', 'FAMILY_SYMPTOMS', 'TWEET_TEXT_PROCESSED']]
    df_test = clean_tweets(df_test,False, remove_duplicates=False)
    print (df_test.shape)
    df = pd.concat([df, df_test])
    
    
    #df = read_pickle1('/home/adhiman/SAR-z/Eda/processed/new/auged_1_2.pkl')

    df =df.sample(frac=1)
    print (df.shape)
    print (df.head())


    df["AUTHOR_OR"] = df["AUTHOR_COVID"] + df["AUTHOR_SYMPTOMS"] > 0
    df["FAMILY_OR"] = df["FAMILY_COVID"] + df["FAMILY_SYMPTOMS"] > 0

    df["AUTHOR_OR"] = df["AUTHOR_OR"].replace(0, False).replace(1,True)
    df["FAMILY_OR"] = df["FAMILY_OR"].replace(0, False).replace(1,True)'''
    
    
    label = label
    text =feat
    
    print (df[label].value_counts())
    #df[label] = df[label].astype(bool)
    
    #print (df[label].value_counts())

    le = preprocessing.LabelEncoder()
    df[label]=le.fit_transform(df[label])





    '''class_size = df[label].sum()
    df_new = pd.concat([
    df[df[label] == 0].sample(int(class_size)),
    df[df[label] == 1]])'''

    #df_new = df
    #print (df[label].value_counts())







    train_text, temp_text, train_labels, temp_labels = train_test_split(df[text], df[label],test_size=0.3, stratify=df[label]) 
    
    #train_text, temp_text, train_labels, temp_labels = train_test_split(df, df[label], test_size=0.3, stratify=df[label]) 


    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                    test_size=0.5, 
                                                                    stratify=temp_labels)
    #print (train_text.shape)
    print (train_labels.value_counts())
    print (val_labels.value_counts())
    print (test_labels.value_counts())
    
    
    
    
    '''print ("DOING AUGMENTATION")
    eda_supplement.prepare4eda(label, train_text)
    print ("here")
    
    new_df = eda_supplement.convert_auged(train_text)
    
    print (train_text.groupby(by=['AUTHOR_OR', 'FAMILY_OR']))
    
    train_text = df[text]
    train_labels = df[label]
    
    print ("DATA SIZE AFTER AUGMENTATION")
    print (train_text.shape)
    print (train_labels.shape)
    
    
    
    val_text = val_text[text]
    test_text = test_text[text]
    
    print ("VAL AND TEST SIZE")
    print (val_text.shape)
    print (test_text.shape)'''
   
    '''class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    print (samples_weight)'''
    
    return train_text, train_labels, val_text, val_labels, test_text, test_labels, df


def data_man(label, feat):
    
    df_train= read_pickle1('/home/adhiman/SAR-z/Eda/processed/new/train-val-test/train.pkl')
    df_val= read_pickle1('/home/adhiman/SAR-z/Eda/processed/new/train-val-test/val.pkl')
    df_test= read_pickle1('/home/adhiman/SAR-z/Eda/processed/new/train-val-test/test.pkl')
    
    '''df_train[label] = list(map(lambda ele: ele == "True", df_train[label].tolist()))
    df_val[label] = list(map(lambda ele: ele == "True", df_val[label].tolist()))
    df_test[label] = list(map(lambda ele: ele == "True", df_test[label].tolist()))'''
    df_train[label] = df_train[label].astype(bool)
    df_test[label] = df_test[label].astype(bool)
    df_val[label] = df_val[label].astype(bool)
    
    df = pd.concat([df_train, df_val, df_test])
    train_text = df_train[feat]
    train_labels = df_train[label]
    
    val_text = df_val[feat]
    val_labels = df_val[label]
    
    test_text = df_test[feat]
    test_labels = df_test[label]
    
    
    return train_text, train_labels, val_text, val_labels, test_text, test_labels, df

    
    
    
def get_data(label, split_type, df, train_text, train_labels, val_text, val_labels, test_text, test_labels, batch_size = 32, feat='TWEET_TEXT_PROCESSED'):
    
    if split_type =='auto':
        train_text, train_labels, val_text, val_labels, test_text, test_labels, df = datasplit(label, feat)
    if split_type =='manual':
        train_text, train_labels, val_text, val_labels, test_text, test_labels, df = data_man(label, feat)
    print ("data split shapes")
    print (train_text.shape)
    print (val_text.shape)
    print (test_text.shape)

    #model_name = 'digitalepidemiologylab/covid-twitter-bert'
    #model_name = 'digitalepidemiologylab/covid-twitter-bert-v2'
    model_name = 'bert-large-uncased'


    #bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
    
    #bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
    bert = AutoModel.from_pretrained('bert-large-uncased')

    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    seq_len = [len(i.split()) for i in train_text]

    #pd.Series(seq_len).hist(bins = 30)



    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = 60, #96
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = 60,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = 60,
        pad_to_max_length=True,
        truncation=True
    )



    ## convert lists to tensors

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    #train_y = train_y.unsqueeze(1)
    print (train_y.shape)

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())
    #val_y = val_y.unsqueeze(1)

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())
    #test_y = test_y.unsqueeze(1)



    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)


    #train_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    
    return (bert,train_dataloader, val_dataloader, test_seq, test_mask,test_y, val_seq, val_mask,val_y, train_labels, df )
    
    
def check_gpu_usage():

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name())
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    
    

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
      
        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # relu activation function
        self.relu =  nn.ReLU(0.1)

        # dense layer 1
        #self.fc1 = nn.Linear(768,512)
        self.fc1 = nn.Linear(1024,512)
        

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,1)

        #softmax activation function
        #self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False) #try False also 

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        

        # apply softmax activation
        #x = self.softmax(x)
        #x = self.sigmoid(x)
        #x = self.tanh(x)
        #print (x)
        return x
  


    # function to train the model
def train(model, optimizer, train_dataloader, cross_entropy):
    model.train()
    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    print (optimizer.param_groups[0]['lr'])
    
    for step,batch in enumerate(train_dataloader):
    
        # progress update after every 50 batches.
        #if step % 10 == 0 and not step == 0:
            #print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask)
    
        labels = labels.to(torch.float32)
        labels = labels.unsqueeze(1)
        #print (preds.shape)
        #print (labels.shape)
        
        
        #print (preds)
        #print (labels)
        
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)
        #loss = loss_fn(preds, labels)
        #loss = criterion(preds, labels)

        
        
        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()
        

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    
    #returns the loss and predictions
    return avg_loss, total_preds



# function for evaluating the model
def evaluate(model, val_dataloader, cross_entropy):
  
    print("\nEvaluating...")
    
    temperature = nn.Parameter(torch.ones(1).cuda())

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
    
        # Progress update every 50 batches.
        if step % 10 == 0 and not step == 0:

            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            labels = labels.to(torch.float32)
            labels = labels.unsqueeze(1)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    #print (total_preds)
    return avg_loss, total_preds


def T_scaling(logits, args):
    temperature = args.get('temperature', None)
    return torch.div(logits, temperature)


def Temp_scaling(val_dataloader,net, label):
    
    import torch.optim as optim
    
    temperature = nn.Parameter(torch.ones(1).cuda())
    args = {'temperature': temperature}
    criterion = nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    logits_list = []
    labels_list = []
    temps = []
    losses = []

    for i, data in enumerate(tqdm(val_dataloader, 0)):
        sent, mask, labels = data[0].to(device), data[1].to(device),  data[2].to(device)

        net.eval()
        with torch.no_grad():
            logits_list.append(net(sent, mask))
            labels_list.append(labels)

    # Create tensors
    logits_list = torch.cat(logits_list).to(device)
    labels_list = torch.cat(labels_list).to(device)

    def _eval():
        loss = criterion(T_scaling(logits_list, args), labels_list)
        loss.backward()
        temps.append(temperature.item())
        #print (loss)
        #print (type(loss))
        #print (loss.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
        return loss


    optimizer.step(_eval)

    print('Final T_scaling factor: {:.2f}'.format(temperature.item()))

    plt.subplot(121)
    plt.plot(list(range(len(temps))), temps)

    plt.subplot(122)
    plt.plot(list(range(len(losses))), losses)
    plt.show()
    plt.savefig("/home/adhiman/SAR-z/classifier/ct_bert/"+str(label)+"_temp_scale.pdf", bbox_inches='tight')
    plt.close()
    
    return temperature


def run(bert, label, train_dataloader, val_dataloader, train_labels, enc_num, save_path):
    
    


    #########################################################


    epochs =150
    my_lr = 2e-5
    enc_num = 0-enc_num


    #######################################################################

    print("=====================================================================================================")    

    #check_gpu_usage()
    torch.cuda.empty_cache() 
    #train_dataloader= train_dataloader.to(device)
    #val_dataloader = val_dataloader.to(device)
    print ("=========================Total layers")
    print (len(list(bert.parameters())))
    

        
    for name, param in list(bert.named_parameters())[:enc_num]:
        param.requires_grad = False
        
    for name, param in bert.named_parameters():
        if param.requires_grad == True:
            print('I am not frozen: {}'.format(name))
        
    '''modules = [L1bb.embeddings, *L1bb.encoder.layer[:5]] #Replace 5 by what you want
    for module in mdoules:
        for param in module.parameters():
            param.requires_grad = False'''
    
    '''for param in bert.parameters():
        param.requires_grad = False ###False for keeping the weights of the pre-trained encoder frozen and optimizing only the weights of the head layers. '''



    #loss_fn = nn.BCELoss()

    #device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    print()

    print(torch.cuda.current_device())

    #torch.cuda.set_device(1) 
    #print(torch.cuda.current_device())

    
    #batch_size
    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(bert)

    model = nn.DataParallel(model)


    # push the model to GPU
    model = model.to(device)


   


    # define the optimizer
    optimizer = AdamW(model.parameters(),
                      lr = my_lr, eps= 1e-08)          # learning rate

    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    scheduler = ReduceLROnPlateau(optimizer, 'min', patience= 4, factor=0.01, verbose=True)






    #compute the class weights
    #class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)





    class_weights = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(train_labels),
                                            y = train_labels                                                    
                                        )
    #class_weights = dict(zip(np.unique(train_labels), class_weights)),
    #class_weights



    #class_weights = [0.52226277, 3.54477612]
    print("Class Weights:",class_weights)

    # converting list of class weights to a tensor
    weights= torch.tensor(class_weights,dtype=torch.float)

    # push to GPU
    weights = weights.to(device)
    if label == 'AUTHOR_OR':
        pos_wt = torch.tensor([2.86])
    if label == 'FAMILY_OR':
        pos_wt = torch.tensor([5.12])
    if label =='ABT_FAMILY':
        pos_wt = torch.tensor([2.82])
        
    #pos_wt = torch.FloatTensor([weights[1]]*32)
    pos_wt = pos_wt.to(device)

    # define the loss function
    #cross_entropy  = nn.NLLLoss(weight=weights) 
    #cross_entropy  = nn.BCELoss(weight=weights) 
    #cross_entropy  = nn.BCEWithLogitsLoss(pos_weight=weights[1])  #pos_weight=weights
    cross_entropy  = nn.BCEWithLogitsLoss(pos_weight=pos_wt) 
    
    # number of training epochs

    #dice loss
    #criterion = SelfAdjDiceLoss()



    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]






    es=0
    #for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        #train model
        train_loss, _ = train(model, optimizer, train_dataloader, cross_entropy)



        #evaluate model
        valid_loss, _ = evaluate(model, val_dataloader, cross_entropy)
        scheduler.step(valid_loss)

         # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.6f}')
        print(f'Validation Loss: {valid_loss:.6f}')
        valid_loss = round(valid_loss, 7)
        train_loss = round(train_loss, 7)
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            es =0
            torch.save(model.state_dict(), save_path)
        else:
            es += 1
            print("Counter {} of 10".format(es))

            if es > 10:
                print("Early stopping with best_valid_loss: ", best_valid_loss, "and valid_loss for this epoch: ", valid_loss, "...")
                break



        #check_gpu_usage()

    return model, train_losses, valid_losses, epoch, optimizer.param_groups[0]['lr'], 


def test(model, test_seq, test_mask, test_y, train_losses, valid_losses, label, df, epochs, my_lr,batch_size, enc_num, train_index,test_index,  train_index2, test_index2,val, val_th=0 ): #, tempr=0
    
    
    #load weights of best model
    #path = '/home/adhiman/SAR-z/classifier/ct_bert/saved_weights_ctbertdnn_2.pt'
    #model.load_state_dict(torch.load(path))

    test_seq = test_seq.to(device)
    test_mask = test_mask.to(device)
    #test_y = test_y.to(device)



    # get predictions for test data
    with torch.no_grad():
        preds = model(test_seq, test_mask)
        preds = preds.detach().cpu().numpy()
    
    '''if tempr!='x':
        pred = T_scaling(preds, tempr)'''

    #print (preds)
    #print (test_y)
    
    pred_sigmoid = torch.sigmoid(torch.from_numpy(preds))
    #pred_tanh = torch.tanh(torch.from_numpy(preds))
    #preds = np.argmax(preds, axis = 1)

    if val==True:
        f_th = 0
        f_f1 = 0
        for th in np.arange(0.2,1.0, 0.001):

            pred_class = pred_sigmoid>th
            #print(classification_report(test_y, pred_class))
            #acc_score =accuracy_score(test_y, pred_class)
            #print("Accuracy of covBERT+DNN is:",acc_score)
            #report = classification_report(test_y, pred_class)
            f1 = f1_score(test_y, pred_class, average ='macro')
            if f_f1 <f1:
                f_th = th
                f_f1 = f1
        pred_class = pred_sigmoid>f_th
        print ("for validation set")
        acc_score =accuracy_score(test_y, pred_class)
        print("Accuracy of covBERT+DNN is:",acc_score)
        print ("Threshold: "+str(f_th))
        return pred_sigmoid, torch.from_numpy(preds), f_th, f_f1
            
    else:
        pred_class = pred_sigmoid>val_th
        print ("for testing set")
        print ("Threshold: "+str(val_th))
        acc_score =accuracy_score(test_y, pred_class)
        print("Accuracy of covBERT+DNN is:",acc_score)
        f1 = f1_score(test_y, pred_class, average ='macro')
        return pred_sigmoid,  torch.from_numpy(preds), val_th, f1
        
        
        
        
        '''lines = report.split('\n')
        textfile = open("/home/adhiman/SAR-z/classifier/ct_bert/results_sigmd_thshld.txt", "a")
        textfile.write("===================================================================================================================" + "\n")
        #textfile.write(str(model)+'\n')

        textfile.write("label: "+ str(label) + "\n")
        textfile.write("threshold: "+ str(th) + "\n")
        textfile.write("train_index1: "+ str(train_index[0]) + "\n")
        textfile.write("test_index1: "+ str(test_index[0]) + "\n")
        textfile.write("train_index2: "+ str(train_index2[0]) + "\n")
        textfile.write("test_index2: "+ str(test_index2[0]) + "\n")
        textfile.write("Using encoders: "+ str(enc_num) + "\n")
        textfile.write("class0: "+ str(df[label].value_counts()) + "\n")

        textfile.write("lr: "+ str(my_lr) + "\n")
        textfile.write("epochs: "+str(epochs)+"\n")
        textfile.write("Bacth size: "+str(batch_size)+"\n")
        for element in lines:
            textfile.write(element + "\n")
        textfile.write("accuracy: "+str(acc_score)+"\n")
        #textfile.write("Temperature: "+str(tempr.item())+"\n")
        textfile.close()


        plt.plot(train_losses, label="Training loss")
        plt.plot(valid_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.savefig("/home/adhiman/SAR-z/classifier/ct_bert/loss_sig_"+str(label)+".pdf", bbox_inches='tight')
        plt.close()'''
        
    #classification_report_csv(report)

    #del model
    #del preds
    #torch.cuda.empty_cache() 

    

def data_fold(X, Y, label, feat,train_index, test_index, sk):
    
    if sk==1:
        temp_text, test_text = X.values[train_index], X.values[test_index]
        temp_labels, test_labels = Y.values[train_index], Y.values[test_index]
    if sk==2:
        temp_text, test_text = X[train_index], X[test_index]
        temp_labels, test_labels = Y[train_index], Y[test_index]
    
    return temp_text, test_text, temp_labels, test_labels


def main2(label,enc_num, run_num, batch_size, feat):
    
    out_file = open("/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/thresholds.txt", "a")
    df0 = read_pickle1('/disks/sdb/adhiman/SAR-z/labelbox itr3/with consensus/data_for_validation/itr123.pkl')
    df0['ABT_FAMILY'] = df0['FAMILY'].astype(bool)
    out_file.write("========================================================================\n")
    out_file.write(label+'\n')
    df0 = clean_tweets(df0,False, remove_duplicates=False)
    skf = StratifiedKFold(n_splits=10)
    vec = []
    le = preprocessing.LabelEncoder()
    df0[label]=le.fit_transform(df0[label])
    X =df0[feat]
    Y = df0[label]
    out_file.write(label+'\n')
    for train_index, test_index in tqdm(skf.split(X, Y)):
        print ("incoming data")
        print (X.shape)
        
        temp_text, test_text, temp_labels, test_labels = data_fold(X, Y, label, feat,train_index, test_index, sk=1 )
        print ("first k fold shape")
        print (temp_text.shape)
        print (test_text.shape)
        print (np.unique(test_labels, return_counts=True))
        out_file.write("*************************************\n")
        out_file.write(label+'\n')
    
        out_file.write(str(test_index[0])+'\n')
        skf2 = StratifiedKFold(n_splits=9)
        f_save_path = ""
        f_val_f1 = 0
            
        for train_index2, test_index2 in skf2.split(temp_text, temp_labels): 
            
            print (test_index[0])
            if os.path.exists('/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/abt_family_val/preds_'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.csv'):
                print ("PATH EXISTS")
                continue
                
            train_text, val_text, train_labels, val_labels = data_fold(temp_text, temp_labels, label, feat,train_index2, test_index2, sk=2 )
            out_file.write(str(train_index2[0])+'\n')
            out_file.write(str(test_index2[0])+'\n')
            
            print ("second k fold shape")
            print (train_text.shape)
            print (val_text.shape)
            print (np.unique(train_labels, return_counts=True))
            print (np.unique(val_labels, return_counts=True))
            
            #bert,train_dataloader, val_dataloader, test_seq, test_mask,test_y, train_labels, df  = get_data(label, 'split_type', df0, train_text, train_labels, val_text, val_labels, test_text, test_labels, batch_size, feat)
            
            bert,train_dataloader, val_dataloader, test_seq, test_mask,test_y, val_seq, val_mask, val_y, train_labels, df  = get_data(label, 'split_type', df0, train_text, train_labels, val_text, val_labels, test_text, test_labels, batch_size, feat)
            
            if label =='AUTHOR_OR':
                save_path ='/disks/sdb/adhiman/SAR-z/ct_bert/testing weight/cross_val_ctbert/saved_weights_ctbertdnn_auth_test'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.pt'
            if label =='FAMILY_OR':
                save_path ='/disks/sdb/adhiman/SAR-z/ct_bert/testing weight/cross_val_ctbert/saved_weights_ctbertdnn_fam_test'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.pt'
            if label =='ABT_FAMILY':
                save_path ='/disks/sdb/adhiman/SAR-z/ct_bert/testing weight/cross_val_ctbert/saved_weights_ctbertdnn_abt_fam_test'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.pt'

            print (save_path)
            model, train_losses, valid_losses, epch, lr = run(bert, label, train_dataloader, val_dataloader, train_labels, enc_num, save_path)
            
            val_preds_sig, val_preds, val_th, val_f1 = test(model, val_seq, val_mask, val_y, train_losses, valid_losses,label, df, epch, lr,batch_size, enc_num, train_index,test_index,  train_index2, test_index2, val =True, val_th = 0 )
            
            df_val = pd.DataFrame(df.values[test_index2],columns =df.columns)
            df_val['Sig_Predictions'] = val_preds_sig
            df_val['Predictions'] = val_preds
            if label =='AUTHOR_OR': 
                df_val.to_csv('/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/author_val2/preds_'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.csv', index =None)
            if label =='FAMILY_OR':
                df_val.to_csv('/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/family_val2/preds_'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.csv', index =None)
            if label =='ABT_FAMILY':
                df_val.to_csv('/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/abt_family_val/preds_'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.csv', index =None)
            '''if val_f1>f_val_f1:
                print ("updating data \n old f1"+str(f_val_f1)+"\t new f1 "+str(val_f1))
                out_file.write("updating data \n old f1"+str(f_val_f1)+"\t new f1 "+str(val_f1))
                f_save_path = save_path
                f_val_f1 = val_f1
                #tempr = Temp_scaling(val_dataloader,model, label)'''

            test_preds_sig, test_preds, test_th, test_f1 = test(model, test_seq, test_mask, test_y, train_losses, valid_losses,label, df, epch, lr,batch_size, enc_num, train_index,test_index,  train_index2, test_index2,  val =False, val_th =val_th  ) #, tempr

            df_temp = pd.DataFrame(df.values[test_index],columns =df.columns)
            df_temp['Predictions'] = test_preds
            df_temp['Sig_Predictions'] = test_preds_sig
            if label =='AUTHOR_OR': 
                df_temp.to_csv('/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/author_test2/preds_'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.csv', index =None)
            if label =='FAMILY_OR':
                df_temp.to_csv('/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/family_test2/preds_'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.csv', index =None)
            if label =='ABT_FAMILY':
                df_val.to_csv('/disks/sdb/adhiman/SAR-z/ct_bert/split_data/ct_bert/abt_family_test/preds_'+str(test_index[0])+'_'+str(train_index2[0])+'_'+str(test_index2[0])+'.csv', index =None)

            out_file.write(str(val_th)+"\n")
            out_file.flush()
            os.fsync(out_file.fileno())
    out_file.close()
            
    
def main(label,enc_num, run_num, batch_size, feat):
    
    bert,train_dataloader, val_dataloader, test_seq,  test_mask,test_y,  train_labels, df = get_data( label,'auto', 0,0,0,0,0, batch_size,feat )
    if label =='AUTHOR_OR':
        save_path ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/cross_val_ctbert/saved_weights_ctbertdnn_auth_test'+str(run_num)+'.pt'
    if label =='FAMILY_OR':
        save_path ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/cross_val_ctbert/saved_weights_ctbertdnn_fam_test'+str(run_num)+'.pt'
    if label =='ABT_FAMILY':
        save_path ='/home/adhiman/SAR-z/classifier/ct_bert/testing weight/cross_val_ctbert/saved_weights_ctbertdnn_abt_fam_test'+str(run_num)+'.pt'
    print (save_path)
    model, train_losses, valid_losses, epch, lr = run(bert, label, train_dataloader, val_dataloader, train_labels, enc_num, save_path)

    #tempr = Temp_scaling(val_dataloader,model, label)

    test(model, test_seq, test_mask, test_y, train_losses, valid_losses,label, df, epch, lr,batch_size, enc_num, 0,0,0,0 ) #, tempr


    
    
if __name__ == "__main__":
    
    args = sys.argv[1:]
    print (args)
    label = args[0]
    enc_num = int(args[1])
    run_num = int(args[2])
    batch_size = 32
    #label = 'FAMILY_OR'
    feat = 'TWEET_TEXT_PROCESSED'
    
    #main(label,enc_num, run_num, batch_size, feat)
    main2(label,enc_num, run_num, batch_size, feat)
    
        
        