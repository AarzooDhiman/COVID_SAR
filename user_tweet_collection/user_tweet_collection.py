#This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).

import datetime, time, os, json, requests, glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from json import JSONDecodeError
import pickle5 as pickle
import sys


USER_ID = "USER_ID"
TWEET_ID = "TWEET_ID"
LOCATION = "LOCATION"
TIMESTAMP = "TIMESTAMP"
TWEET_TEXT = "TWEET_TEXT" 

columns = [TWEET_ID, USER_ID, LOCATION, TIMESTAMP, TWEET_TEXT]

user_columns = ["id", "created_at", "description", "location","protected", "followers_count", "following_count", "tweet_count", "listed_count"]


headers = {'authorization': "BEARER TOKEN"} #add your token here

url = "https://api.twitter.com/2/users" # url for getting user timeline
querystring1 = {"ids":None,"user.fields":"id,created_at,location,protected,description,public_metrics"} #, "start_time":"2020-01-01T00:00:00Z"

def make_api_query(ids):
    #making API request
    querystring1["ids"] = ",".join([str(x) for x in ids])
    #print (querystring1)
    response_text = requests.request("GET", url, headers=headers, params=querystring1).text
    if len(response_text) == 0:
        print("No data received")
        return None
    #print (response_text)
    response = json.loads(response_text)
    if "data" not in response:
        if "errors" in response:
            print(response["errors"])
            #print(response)
        return None
    return response


def read_pickle1(pth):
    with open(pth, "rb") as fh:
        data = pickle.load(fh)
    return (data)


def parse_user(user):
    # parsing the json response
    timestamp = user.get("created_at")
    user_id = user.get("id")
    bio = user.get("description")
    geo = user.get("location")
    protected = user.get("protected")
    metrics = user["public_metrics"]
    followers_count = metrics.get("followers_count")
    following_count = metrics.get("following_count")
    tweet_count = metrics.get("tweet_count")
    listed_count = metrics.get("listed_count")
    return user_id, timestamp, bio, geo, protected, followers_count, following_count, tweet_count, listed_count

def parse_response(j_response):
    parsed_data = map(parse_user, j_response["data"])
    temp_df = pd.DataFrame(parsed_data, columns = user_columns)
    temp_df = temp_df.set_index("id", drop=False)
    return temp_df


def download_and_parse(ids):
    #print (ids)
    response = make_api_query(ids)
    if response is None:
        return None
    return parse_response(response)


headers = {'authorization': "BEARER TOKEN"} #add your token here
      
          
##change
querystring2 = {"since_id":1,"exclude":"retweets","tweet.fields":"id,author_id,created_at,geo","max_results":100, "end_time":"2022-02-01T00:00:00Z"} #TODO: limit date download 

def make_timeline_query(user_id, next_token=None):
    # design query for user timeline
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    
    if next_token is None:
        querystring2.pop("pagination_token", None)
    else:
        querystring2["pagination_token"] = next_token
        
    response_text = requests.request("GET", url, headers=headers, params=querystring2).text
    #print (response_text)
    if len(response_text) < 3:
        print("No data received")
        return None
    
    try:
        response = json.loads(response_text)
    except JSONDecodeError as e:
        print ("hitting error 1")
        print(response_text)
        print(e)
        return None
    
    if "data" not in response:
        if "errors" in response:
            print ("hitting error 2")
            print(response["errors"])
            raise Exception(str(response["errors"]))
        print ("hitting error 3")
        print(response)
        return None
    return response


def get_user_timeline(user_id):
    response = make_timeline_query(user_id)
    #print (response)
    if response is None:
        raise Exception("No Tweets returned")
    timeline = response["data"]

    while "next_token" in response["meta"]:
        response = make_timeline_query(user_id, response["meta"]["next_token"])
        if response is None:
            break
        timeline.extend(response["data"])
    return timeline

def parse_tweet(status):
    #json_str = json.dumps(status._json)
    #parsed = json.loads(json_str)
    parsed = status
    timestamp = parsed["created_at"]
    tweet_id = parsed["id"]
    text = parsed["text"]
    user_id = parsed["author_id"]
    geo = parsed.get("geo", None)
    return tweet_id, user_id, geo, timestamp, text


statuses_columns = ["user_id", "status", "size"]

def status_update(user_id, status, size):
    statuses = pd.read_csv(f"../data/seed_tweets/statuses.csv")
    statuses = statuses.append(pd.Series([user_id, status, size], index=statuses_columns), ignore_index=True)
    #print (statuses)
    #statuses.to_csv(f"/disks/sdb/adhiman/SAR_data/data/seed_tweets/statuses.csv", index=False, columns = statuses_columns)


# method for getting user profiles
def getting_uk_users(source, dest):

    month = source.split('/')[-1] ###remove
    print ("====================================================================================")
    print (month)
    data = read_pickle1(source)
    
    print (data.head())
    # getting the user iDs for profile collection
    downloaded_users = pd.DataFrame(data['USER_ID'].unique(), columns = ['USER_ID'])
    print ("===========shape of users=========")
    print (downloaded_users.shape)
    print (downloaded_users.head())
    downloaded_df =pd.DataFrame()

    
    
    #######################users to download
    if not os.path.exists(dest + "/all_user_profiles"):
        os.mkdir(dest + "/all_user_profiles")
    
    limit = 100            
    for i in range(0, downloaded_users.shape[0], limit):
        print (i)
        #getting 100 users at a time
        if i+5<downloaded_users.shape[0]:
            # getting profiles
            try:
                downloaded_df0 = download_and_parse(downloaded_users['USER_ID'][i: i+limit])
            except:
                pass
        else:
            try:
                downloaded_df0 = download_and_parse(downloaded_users['USER_ID'][i:downloaded_df.shape[0]])
            except:
                pass
        downloaded_df = pd.concat([downloaded_df, downloaded_df0])
        print (downloaded_df.shape)
        #print ('==========downloaded df=============')
        #print (downloaded_df)
        downloaded_df.to_pickle(dest + "/all_user_profiles/"+str(month))  
        
        time.sleep(3)


    print ('==========downloaded df=============')
    print (downloaded_df.head())
    
    downloaded_df.to_pickle(dest + "/all_user_profiles/"+str(month)) ###remove
    
def collect_timeline(source, dest):
    
    print ("==================================================================================================================\n")
    month =source.split('/')[-1]
    print (source)
    print (month)
    downloaded_users = read_pickle1(source)
    
    print (downloaded_users.head())

    users = downloaded_users[downloaded_users.UK]["id"].astype(str).unique() ### for all UK users
    
    
    CONTINUE = True
    
    print ("USERS in given month")
    print (len(users))
   
    
    # finding existing users so no data collection repetition
    ex_users = glob.glob('../data/user_timelines/*.pkl')
    existing=[e.split('/')[-1].replace('.pkl','') for e in ex_users]
    print (existing[:5])
    rem_users = list(set(users).difference(set(existing))) # using only remaining users
    #rem_users = list(set(users).intersection(set(existing)))
    print ('existing users \t'+str(len(existing)))
    print ('Total users \t'+str(len(users)))
    print ('matching users \t'+str(len(rem_users)))
    print ('starting collections')
    
    
    
    
    with tqdm(total=len(users), position=0, leave=True) as pbar2:
        #for user_id in rem_users:
        for user_id in users:
            if str(user_id) not in existing:
                try:
                    timeline = get_user_timeline(str(user_id))
                    print (len(timeline))
                except Exception as e:
                    print(f"some issue with user: {user_id}")
                    print(e)
                    status_update(user_id, str(e), 0)
                    time.sleep(2)
                    pbar2.update(1)
                    continue
                
                    
                status_update(user_id, "OK", len(timeline)) #file read/save
                #print (timeline)
                #parse JSON and save
                parsed_data = map(parse_tweet, timeline)
                temp_df = pd.DataFrame(parsed_data, columns=columns)
                mnth=month.replace('.pkl','')
                #temp_df.to_pickle(dest+str(mnth)+'/'+str(user_id)+".pkl") #file save
                if user_id not in rem_users:
                    temp_df.to_pickle(dest+str(user_id)+".pkl") #file save
                else:
                    print (user_id)
                    user_exist_data = read_pickle1(dest+str(user_id)+".pkl")
                    temp_df = pd.concat([user_exist_data, temp_df])
                    temp_df.to_pickle(dest+str(user_id)+".pkl") #file save
                time.sleep(5)
                pbar2.update(1)
                
            else:
                print ('already existing')
                pbar2.update(1)
                continue
    


def main():
    
    args = sys.argv[1:]
    print (args)
    #source  = args[0]
    #dest = args[1] 
    
    if args[0]==str(1): # for getting user profile information
        if os.path.isdir(args[1]):
            files = glob.glob(args[1]+'/*pkl')
            for file in files:
                getting_uk_users(file, args[2])
        else:
            print ("getting UK users")
            getting_uk_users(args[1], args[2])
    elif args[0]==str(2): # for getting user timelines
        if os.path.isdir(args[1]):
            files = glob.glob(args[1]+'/*pkl')
            print ("collecting UK users timeline")
            for file in files:
                collect_timeline(file, args[2])
        else:
            print ("collecting UK users timeline")
            collect_timeline(args[1], args[2])

    
    

if __name__ == '__main__':
    
    print ("===================================================================================================================================================================")

    main()



