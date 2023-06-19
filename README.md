# SAR_data

## Labeled data

The classifier has been trained on a dataset of [labeled_tweet_id2](labeled_tweet_id2.txt), which is provided in the file labeled_tweet_id2.txt. This file contains only the Tweet IDs, which can be used to retrieve the corresponding tweets. The dataset includes various features such as the timestamp, tweet ID, user ID, geolocation, and tweet text, which were used to train the classifier.

## Text Preprocessing

The [Preprocess.py](preprocess.py) script is utilized for preprocessing the tweet text before it is used for classification. The preprocessed text is then fed into the classification model as input.

`python preprocess.py`

## Classification

We utilized the Covid-Twitter-BERT (CT-BERT) model for classification, which is a variant of the BERT large uncased model fine-tuned on a dataset of 160 million tweets collected between January 12 and April 16, 2020. We further fine-tuned the CT-BERT model on our labeled tweet dataset and saved the best performing model from our training process.

To run the [ct-bert2-crossval.py](ct-bert2-crossval.py) script, please use the [run_ct_bert.sh](run_ct_bert.sh) shell file.

`sh ./run_ct_bert.sh`

## User Tweets Classification 
After being trained on the labeled tweet dataset, the classification model is then utilized to generate prediction scores for all the tweets of all users. This involves running the model on the entire tweet corpus to classify each tweet based on its content and generate a corresponding prediction score.

`python ctbertppln.py`

## SAR Calculation 

The final estimation of SAR (Secondary Attack Rate) is performed in the [user_class.py](user_class.py) code file. To run the SAR estimation process, please use the provided shell file [run_user_class.sh](run_user_class.sh).

`sh ./run_user_class.sh`
