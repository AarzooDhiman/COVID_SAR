import datetime, time, os, json, requests
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import glob
import pickle5 as pickle



columns = ["TIMESTAMP", "TWEET_ID", "USER_ID", "GEO", "TWEET_TEXT"]
statuses_columns = ["start_date", "end_date", "query", "count", "is_next", "next_token", "last_update"]



tweet_query_text = "(\"i have covid\" OR \"i might have covid\" OR \"i had covid\" OR \"ve got covid\" OR \"i got covid\" OR \"ve had covid\" OR \"i tested positive for covid\" OR \"ve tested positive for covid\" OR \"ve been tested positive for covid\" OR \"I been tested positive for covid\" OR \"i have corona\" OR \"i might have corona\" OR \"i had corona\" OR \"ve got corona\" OR \"i got corona\" OR \"ve had corona\" OR \"i tested positive for corona\" OR \"ve tested positive for corona\" OR \"ve been tested positive for corona\" OR \"I lost my sense of smell\" OR \"ve lost my sense of smell\" OR \"I lost my sense of taste\" OR \"ve lost my sense of taste\" OR \"I lost sense of smell\" OR \"ve lost sense of smell\" OR \"I lost sense of taste\" OR \"ve lost sense of taste\" OR \"I lost the sense of smell\" OR \"ve lost the sense of smell\" OR \"I lost the sense of taste\" OR \"ve lost the sense of taste\" OR \"I lost taste and smell\" OR \"ve lost taste and smell\" OR \"I lost smell and taste\" OR \"ve lost smell and taste\") lang:en -is:retweet"

tweet_query_text_family = "(\"husband\" OR \"wife\" OR \"partner\" OR \"daughter\" OR \"son\" OR \"mum\" OR \"mom\" OR \"mommy\" OR \"dad\" OR \"parent\" OR \"mate\" OR \"boyfriend\" OR \"girlfriend\" OR \"kid\" OR \"child\") (\"has covid\" OR \"might have covid\" OR \"had covid\" OR \"got covid\" OR \"tested positive for covid\" OR \"has corona\" OR \"might have corona\" OR \"had corona\" OR \"got corona\" OR \"tested positive for corona\" OR \"has coronavirus\" OR \"might have coronavirus\" OR \"had coronavirus\" OR \"got coronavirus\" OR \"tested positive for coronavirus\" OR \"sense of smell\" OR \"sense of taste\" OR \"lost smell\" OR \"lost taste\") lang:en -is:retweet"

tweet_query_text_family_loc = "(\"husband\" OR \"wife\" OR \"partner\" OR \"daughter\" OR \"son\" OR \"mum\" OR \"mom\" OR \"mommy\" OR \"dad\" OR \"parent\" OR \"mate\" OR \"boyfriend\" OR \"girlfriend\" OR \"kid\" OR \"child\") (\"has covid\" OR \"might have covid\" OR \"had covid\" OR \"got covid\" OR \"tested positive for covid\" OR \"has corona\" OR \"might have corona\" OR \"had corona\" OR \"got corona\" OR \"tested positive for corona\" OR \"has coronavirus\" OR \"might have coronavirus\" OR \"had coronavirus\" OR \"got coronavirus\" OR \"tested positive for coronavirus\" OR \"sense of smell\" OR \"sense of taste\" OR \"lost smell\" OR \"lost taste\") OR place:\"Texas, USA\" lang:en -is:retweet"

#OR place:\"Texas, USA\" OR (profile_locality:Texas profile_region:USA )))
'''
tweet_query_text_family="(\"husband\" OR \"wife\" OR \"partner\" OR \"daughter\" OR \"son\" OR \"mum\" OR \"mom\" OR \"mommy\" OR \"dad\" OR \"parent\" OR \"mate\" OR \"boyfriend\" OR \"girlfriend\" OR \"kid\" OR \"child\") (\"has covid\" OR \"might have covid\" OR \"had covid\" OR \"got covid\" OR \"tested positive for covid\" OR \"has corona\" OR \"might have corona\" OR \"had corona\" OR \"got corona\" OR \"tested positive for corona\" OR \"has coronavirus\" OR \"might have coronavirus\" OR \"had coronavirus\" OR \"got coronavirus\" OR \"tested positive for coronavirus\" OR \"sense of smell\" OR \"sense of taste\" OR \"lost smell\" OR \"lost taste\") place:e0060cda70f5f341 lang:en -is:retweet"
'''


headers = {'authorization': "BEARER_TOKEN"} #use your bearer token here


url = "https://api.twitter.com/2/tweets/search/all"


def read_pickle1(pth):
    with open(pth, "rb") as fh:
        data = pickle.load(fh)
    return (data)



def make_api_query(query_content):
    response_text = requests.request("GET", url, headers=headers, params=query_content).text
    response = json.loads(response_text)
    #with open('../../../../disks/sda/adhiman/SAR-z/data_withlocation/data_loc.json', 'a', encoding='utf-8') as f:
        #json.dump(response, f, ensure_ascii=False, indent=4)
    if "meta" not in response or response["meta"]["result_count"] == 0: #sometimes the API returns nothing the first time around
        time.sleep(1)
        response_text = requests.request("GET", url, headers=headers, params=query_content).text
        response = json.loads(response_text)
        if response == None:
            print ('nothing')
        #print (response)
        if "meta" not in response or response["meta"]["result_count"] == 0:
            return None
    return response

def get_bucket_range():
    temp = pd.read_csv(f"{tmp_dir}/statuses.csv")
    return list(np.array(temp.to_dict('split')["data"])[:,0])


def status_update(start_date, count, is_next, next_token):
    statuses = pd.read_csv(f"{tmp_dir}/statuses.csv")
    statuses.loc[statuses["start_date"]==start_date,"count"] += count
    statuses.loc[statuses["start_date"]==start_date,"is_next"] = is_next
    statuses.loc[statuses["start_date"]==start_date,"next_token"] = next_token
    statuses.loc[statuses["start_date"]==start_date,"last_update"] = time.strftime("%Y%m%d-%H%M%S")
    statuses.to_csv(f"{tmp_dir}/statuses.csv", index=False, columns = statuses_columns)

def get_next_token_by_date(start_date):
    temp = pd.read_csv(f"{tmp_dir}/statuses.csv")
    return temp[temp["start_date"]==start_date].to_dict('records')[0]


def build_query(start_date):
    row = get_next_token_by_date(start_date)
    query_string = {"tweet.fields":"created_at,lang,geo,author_id", "user.fields":"id,created_at,location,description,public_metrics", "max_results":"500"}
    query_string["start_time"] = row["start_date"]
    query_string["end_time"] = row["end_date"]
    query_string["query"] = row["query"]
    
    if row["is_next"]:
        if row["count"] == 0:
            return query_string
        query_string["next_token"] = row["next_token"]
        return query_string
    return None

def parse_tweet(tweet):
    timestamp = tweet["created_at"]
    tweet_id = tweet["id"]
    text = tweet["text"]
    user_id = tweet["author_id"]
    if "geo" in tweet:
        geo = tweet["geo"] #TODO: fix location!!
    else:
        geo=None
    return timestamp, tweet_id, user_id, geo, text

def parse_response(j_response):
    parsed_data = map(parse_tweet, j_response["data"])
    temp_df = pd.DataFrame(parsed_data, columns = ["TIMESTAMP", "TWEET_ID", "USER_ID", "GEO", "TWEET_TEXT"])
    count = j_response["meta"]["result_count"]
    if "next_token" in j_response["meta"]:
        next_token = j_response["meta"]["next_token"]
    else:
        next_token = None
    return temp_df, count, next_token



if __name__ == '__main__':

    CONTINUE = False
    continue_path = ""


    if CONTINUE:
        tmp_dir = continue_path
        timestr = os.path.basename(continue_path)
    else:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        tmp_dir = f"../data/seed_tweets/" 
        #os.mkdir(tmp_dir)
    print(tmp_dir)


    if not CONTINUE:
        statuses = pd.DataFrame(data=[],columns=statuses_columns)
        date_0="4/1/22" #use the start date format month/date/year
        date_31="4/2/22" #use the end date format month/date/year

        jan_start = datetime.datetime.strptime(date_0, "%m/%d/%y")
        jan_end = datetime.datetime.strptime(date_31, "%m/%d/%y")
        for i in range(1):
            month_start = jan_start + relativedelta(months=i)
            month_end = jan_end + relativedelta(months=i)
            month_start_str = month_start.replace(microsecond=0).isoformat() + "Z"
            month_end_str = month_end.replace(microsecond=0).isoformat() + "Z"
            statuses = statuses.append(pd.Series([month_start_str, month_end_str, tweet_query_text, 0, True, None, None], index=statuses_columns), ignore_index=True)
        statuses.to_csv(f"{tmp_dir}/statuses.csv", index=False, columns = statuses_columns)
        for bucket in get_bucket_range():
            temp_df = pd.DataFrame(columns = ["TIMESTAMP", "TWEET_ID", "USER_ID", "GEO", "TWEET_TEXT"])
            temp_df.to_pickle(f"{tmp_dir}/{bucket}.pkl")

            
    print ('here')
    print (len(get_bucket_range())) 
    
    missing = 0
    while missing<len(get_bucket_range()): 
        missing = 0
        for start_date in get_bucket_range():
            query = build_query(start_date)
            #print (query)
            if query is None:
                missing =missing+ 1
                continue
            json_response = make_api_query(query)
            #print (json_response)
            if json_response is None:
                missing += 1
                print(f"{start_date} {get_next_token_by_date(start_date)['end_date']} 0")
                continue

            tweets, count, next_token = parse_response(json_response)
            temp_df = pd.read_pickle(f"{tmp_dir}/{start_date}.pkl")
            temp_df = temp_df.append(tweets, ignore_index=True)
            temp_df.to_pickle(f"{tmp_dir}/{start_date}.pkl")

            status_update(start_date, count, next_token != None, next_token)
            print(f"{start_date} {count}")

            time.sleep(3)
    
