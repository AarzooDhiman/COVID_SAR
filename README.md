# SAR_data

## Labeled data
The classifier has been trained on the labeled data given in file [labeled_tweet_id2](labeled_tweet_id2.txt)
The file contains only the Tweet IDs. The tweets can be retrieved using these Tweet IDs. The features used in our work are TIMESTAMP, TWEET_ID, USER_ID, GEO, TWEET_TEXT

## Text Preprocessing
[Preprocess](preprocess.py) is used for pre-processing the tweet text. This preprocessed text is used for the classification. 

`python preprocess.py`

## Classification
For classification we use Covid-Twitter-BERT, BERT large uncased model fine tuned on 160M tweets collected between January 12 and April 16, 2020. We further fine-tuned the model on our labeled tweets data. We save the best performing model from this code. 

To run [ct-bert2-crossval.py](ct-bert2-crossval.py) use the shell file [run_ct_bert.sh](run_ct_bert.sh)

`sh ./run_ct_bert.sh`

## User Tweets Classification 
The trained model is further run on tweets of all the users to generate the prediction scores for all the tweets of all the users. 

`python ctbertppln.py`

## SAR Calculation 
Final SAR estimation is done in code file [user_class.py](user_class.py)
To run this SAR estimation use the shell file [run_user_class.sh](run_user_class.sh)

`sh ./run_user_class.sh`
