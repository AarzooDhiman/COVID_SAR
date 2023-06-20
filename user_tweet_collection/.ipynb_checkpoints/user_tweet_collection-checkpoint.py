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


headers = {'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAAHsoMgEAAAAAN4KZWup8keAkX5fNoeHu8qHCB7Y%3DbA7EbQYr03L5g9Rs471pqvqpFpbI5jTeHwnZeAYP1achIBFIt3"} #tomasz
#headers = {'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAAHF2WAEAAAAAhIErrd34%2BshvMV9CAnU2dydk7lM%3D1APWEjTU8qrSwXkmhuL9L4RiQ4S7QGUeK6ktNqmYr8TcwdhVjd"} #bill
#headers = {'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAADr8XwßEAAAAAHVCYdwhoUTIEmy%2BevrzvYmBpNrU%3DFAorOislCBNrGOJnfamL80Specc1zdFGyYNKqAi0bHDDMnqTOC"} #ing1
#headers ={'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAADxHYwEAAAAAjPiJAPulgIw4bJz9FbkjN9exsFc%3DzazxkQRmOaEac0Pa79NkmWWQ7h7HTiQFliJ93CADiYoAHyhynn"} #ing2
#headers ={'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAAA9nXQEAAAAADJevKM8eoZ6yNdlHiHwl09rXir4%3Dh8bszwJCXxpYpQqR2KGrFhJcpP1qTv56Gvpcwsa0gM1xRTrLtM"} #vm

url = "https://api.twitter.com/2/users"
querystring1 = {"ids":None,"user.fields":"id,created_at,location,protected,description,public_metrics"} #, "start_time":"2020-01-01T00:00:00Z"

def make_api_query(ids):
    #print (ids)
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


headers = {'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAAHsoMgEAAAAAN4KZWup8keAkX5fNoeHu8qHCB7Y%3DbA7EbQYr03L5g9Rs471pqvqpFpbI5jTeHwnZeAYP1achIBFIt3"} #tomaz
#headers = {'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAAHF2WAEAAAAAhIErrd34%2BshvMV9CAnU2dydk7lM%3D1APWEjTU8qrSwXkmhuL9L4RiQ4S7QGUeK6ktNqmYr8TcwdhVjd"} #bill
#headers = {'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAADr8XwEAAAAAHVCYdwhoUTIEmy%2BevrzvYmBpNrU%3DFAorOislCBNrGOJnfamL80Specc1zdFGyYNKqAi0bHDDMnqTOC"} #ing1
#headers ={'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAADxHYwEAAAAAjPiJAPulgIw4bJz9FbkjN9exsFc%3DzazxkQRmOaEac0Pa79NkmWWQ7h7HTiQFliJ93CADiYoAHyhynn"} #ing2

#headers ={'authorization': "Bearer AAAAAAAAAAAAAAAAAAAAAA9nXQEAAAAADJevKM8eoZ6yNdlHiHwl09rXir4%3Dh8bszwJCXxpYpQqR2KGrFhJcpP1qTv56Gvpcwsa0gM1xRTrLtM"} #vm          
          
          
##change
querystring2 = {"since_id":1,"exclude":"retweets","tweet.fields":"id,author_id,created_at,geo","max_results":100, "end_time":"2022-02-01T00:00:00Z"} #TODO: limit date download 

def make_timeline_query(user_id, next_token=None):
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
    statuses = pd.read_csv(f"/disks/sda/adhiman/SAR-z/data_oct/statuses.csv")
    statuses = statuses.append(pd.Series([user_id, status, size], index=statuses_columns), ignore_index=True)
    statuses.to_csv(f"/disks/sda/adhiman/SAR-z/data_oct/statuses.csv", index=False, columns = statuses_columns)


'''    
users_list_csv = read_pickle1(f"../../../../disks/sda/adhiman/SAR-z/data/users_to_download.pkl")

users = users_list_csv['USER_ID'].tolist()




folder_path = "../../../../disks/sda/adhiman/SAR-z/data"
existing_db_path = "../../../../disks/sda/adhiman/SAR-z/data/users.pkl"


users_list_csv = read_pickle1(f"{folder_path}/users_to_download.pkl")
print ('users_to_dowload')
print (users_list_csv.shape)





if not os.path.isfile(f"../../../../disks/sda/adhiman/SAR-z/data/users.pkl"):
    users_list_csv = read_pickle1(f"../../../../disks/sda/adhiman/SAR-z/data/users_to_download.pkl")
    users_list_csv["USER_ID"] = users_list_csv["USER_ID"].astype(str)
    users_db = read_pickle1('../../../../disks/sda/adhiman/SAR-z/data_june_sept/user_timelines/20211004-075211/users.pkl')
    downloaded_users = users_list_csv.set_index("USER_ID").join(users_db)
    downloaded_users["id"] = downloaded_users.index
    downloaded_users.to_pickle(f"../../../../disks/sda/adhiman/SAR-z/data/users.pkl")
    downloaded_users.to_csv(f"../../../../disks/sda/adhiman/SAR-z/data/users.csv")

print ('users updated')
print (downloaded_users.head())



temp = read_pickle1(f"../../../../disks/sda/adhiman/SAR-z/data/users_to_download.pkl")
#pbar = tqdm(total=temp[temp["created_at"].isnull()].shape[0], position=0, leave=True)
pbar = tqdm(total=temp.shape[0], position=0, leave=True)
while True:
    downloaded_users = read_pickle1(f"../../../../disks/sda/adhiman/SAR-z/data/users_to_download.pkl")
    #if downloaded_users[downloaded_users["created_at"].isnull()].shape[0] == 0:
        #break
    #current_to_download = list(downloaded_users[downloaded_users["created_at"].isnull()].sample(100)["id"])
    downloaded_df = download_and_parse(downloaded_users['USER_ID'][:5])
    
    print ('==========downloaded df=============')
    print (downloaded_df)
    if downloaded_df is None:
        time.sleep(3)
        continue
    print (downloaded_df)
    downloaded_users.loc[downloaded_df.index, user_columns] = downloaded_df[user_columns]
    downloaded_users.to_pickle(f"../../../../disks/sda/adhiman/SAR-z/data/all_user_profiles.pkl")
    downloaded_users.to_csv(f"../../../../disks/sda/adhiman/SAR-z/data/all_user_profiles.pkl")

    time.sleep(3)
    pbar.update(downloaded_df.shape[0])
pbar.close

downloaded_users =downloaded_users.drop_duplicates(subset=['id'])
'''

def getting_uk_users(source, dest):

    month = source.split('/')[-1] ###remove
    print ("====================================================================================")
    print (month)
    data = read_pickle1(source)
    
    print (data.head())
    downloaded_users = pd.DataFrame(data['USER_ID'].unique(), columns = ['USER_ID'])
    print ("===========shape of users=========")
    print (downloaded_users.shape)
    print (downloaded_users.head())
    downloaded_df =pd.DataFrame()

    
    
    #######################users to download
    if not os.path.exists(dest + "/all_user_profiles"):
        os.mkdir(dest + "/all_user_profiles")
    
        
    '''pr_users = pd.read_csv('/disks/sdb/adhiman/SAR-z/ct_SAR_plots4/p_r.csv')
    downloaded_users =pd.DataFrame(pr_users['USER ID'].unique(), columns = ['USER_ID'])
    downloaded_df =pd.DataFrame()'''
    
    limit = 100            
    for i in range(0, downloaded_users.shape[0], limit):
        print (i)
        #print (downloaded_users['USER_ID'][i: i+5])
        if i+5<downloaded_users.shape[0]:
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
        downloaded_df.to_pickle(dest + "/all_user_profiles/"+str(month))  ###remove
        #downloaded_df.to_csv('/disks/sda/adhiman/SAR-z/raw_tweets/uk_user_description.csv', index= None)
        time.sleep(3)


    print ('==========downloaded df=============')
    print (downloaded_df.head())
    
    downloaded_df.to_pickle(dest + "/all_user_profiles/"+str(month)) ###remove
    #downloaded_df.to_csv('/disks/sda/adhiman/SAR-z/raw_tweets/uk_user_description.csv', index= None)
    
    
    

def check_time(user_ids):
    new_users = []
    for u in user_ids:
        if os.path.exists(f'/disks/sda/adhiman/SAR-z/all_user_timelines/{u}.pkl'):
            avail = read_pickle1(f'/disks/sda/adhiman/SAR-z/all_user_timelines/{u}.pkl')
            
            avail['TIMESTAMP'] = pd.to_datetime(avail.TIMESTAMP)
            avail = avail.sort_values("TIMESTAMP", ascending=False)
            if avail.tail(1).TIMESTAMP.values[0] > np.datetime64('2021-10-30T00:00:00.000000000'):##change
                new_users.append(u)
            if os.path.exists(f'/disks/sda/adhiman/SAR-z/new_user_timelines/{u}.pkl'):
                olddwnld = glob.glob('/disks/sda/adhiman/SAR-z/new_user_timelines/'+str(u)+'_*.pkl') + ['/disks/sda/adhiman/SAR-z/new_user_timelines/'+str(u)+'.pkl']
                for od in olddwnld:
                    od_df = read_pickle1(od)
                    od_df['TIMESTAMP'] = pd.to_datetime(od_df.TIMESTAMP)
                    od_df = od_df.sort_values("TIMESTAMP", ascending=False)
                    if od_df.head(1).TIMESTAMP.values[0] < np.datetime64('2022-01-30T00:00:00.000000000'):##change
                        #print (od_df.head(1).TIMESTAMP.values[0] )
                        #print (od_df)
                        if u not in new_users:
                            #print ("adding new user data")
                            new_users.append(u)

    print ("USERS to start with")
    print (len(new_users))
    print (new_users[:5])
    return new_users
    
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
    #users = check_time(users)
    
    

    '''
    with open('all_users.txt', 'r') as au:
        available_users= au.readlines()

    available_users = [au.strip() for au in available_users]
    print ("users extracted")






    if CONTINUE: 
        processed = pd.read_csv(f"{tmp_dir}/statuses.csv")
        statuses = processed[processed.status == "OK"]
        statuses.to_csv(f"{tmp_dir}/statuses.csv", index=False)
        processed = set(statuses["user_id"])
        users = list(set(users).difference(processed))
    else:
        statuses = pd.DataFrame(data=[],columns=statuses_columns)
        statuses.to_csv(f"{tmp_dir}/statuses.csv", index=False, columns = statuses_columns)



    users = users_list_csv.USER_ID.tolist()
    '''
    
    '''if os.path.exists('/disks/sda/adhiman/SAR-z/data_june_sept/user_timelines/20211004-075211/'+str(user_id)+'.pkl'):
                print ('already exists1')
                pbar2.update(1)
                continue
            if os.path.exists('/disks/sda/adhiman/SAR-z/data_may_allprocessed/user_data/'+str(user_id)):
                print ('already exists2')
                pbar2.update(1)
                continue
            if os.path.exists('/disks/sda/adhiman/SAR-z/data_oct/user_info_auth/user_timelines/'+str(user_id)+'.pkl'):
                print ('already exists2')
                pbar2.update(1)
                continue
            if os.path.exists('/disks/sda/adhiman/SAR-z/data_oct/users_timelines_fam/'+str(user_id)+'.pkl'):
                print ('already exists2')
                pbar2.update(1)
                continue'''
    #random.shuffle(users)
    #users = ['1003390725403369472', '1000015700881281029', '1000445915428212738', '1006661780322570240']
    
    
    #ex_users = read_pickle1('/disks/sda/adhiman/SAR-z/raw_tweets/existing_users.pkl')
    #existing = ex_users.existing.tolist()
    
    
    ex_users = glob.glob('/disks/sdb/adhiman/SAR_data/data/user_timelines/*.pkl')
    existing=[e.split('/')[-1].replace('.pkl','') for e in ex_users]
    print (existing[:5])
    #rem_users = list(set(users).difference(set(existing)))
    rem_users = list(set(users).intersection(set(existing)))
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
                except Exception as e:
                    print(f"some issue with user: {user_id}")
                    print(e)
                    status_update(user_id, str(e), 0)
                    time.sleep(2)
                    pbar2.update(1)
                    continue
                
                
                #new_user = {'existing': str(user_id)}
                #ex_users.loc[ex_users.shape[0]] = new_user
                #added=pd.DataFrame()
                #added = pd.DataFrame(list(str(user_id)), columns =['existing'])
                #ex_users= pd.concat([ex_users,added])
                #ex_users.to_pickle('/disks/sda/adhiman/SAR-z/raw_tweets/existing_users.pkl')    
                    
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
    
    if args[0]==str(1):
        if os.path.isdir(args[1]):
            files = glob.glob(args[1]+'/*pkl')
            for file in files:
                getting_uk_users(file, args[2])
        else:
            print ("getting UK users")
            getting_uk_users(args[1], args[2])
    elif args[0]==str(2):
        if os.path.isdir(args[1]):
            files = glob.glob(args[1]+'/*pkl')
            print ("collecting UK users timeline")
            for file in files:
                collect_timeline(file, args[2])
        else:
            print ("collecting UK users timeline")
            collect_timeline(args[1], args[2])
        
        '''files = glob.glob(args[1]+'/*.pkl')
        for file in files:
            collect_timeline(file, args[2])
        collect_timeline(args[1], args[2])'''
    
    

if __name__ == '__main__':
    
    print ("===================================================================================================================================================================")

    main()



