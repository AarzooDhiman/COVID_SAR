import pandas as pd
## preparing the dataset
import pickle5 as pickle
import pandas as pd
import glob
from multiprocessing import Pool
import os
from fuzzywuzzy import fuzz


def read_pickle1(pth):
    with open(pth, "rb") as fh:
        fh.seek(0)
        data = pickle.load(fh)
    return (data)

def get_uk(path):
    loc = pd.read_csv(path)
    loc = loc.sort_values(by=['2022'], ascending=False)
    loc.head()
    uk_locs = loc['name'].tolist()[:20]
    uk_locs = [loc.replace('\xa0', '') for loc in uk_locs]
    #adding some more common names used for UK
    uk_locs = uk_locs+['England', 'United Kingdom', 'UK', 'Greater London', 'Greater Manchester','Scotland', 'Wales', 'Northern Ireland', 'Newport', 'Belfast', 'Derry']
    #print (uk_locs)
    return uk_locs

def get_uk_us(path, uk_loc):
    us_loc = pd.read_csv(path, encoding='latin-1')
    us_locs = us_loc.City.tolist()
    #print (us_locs)
    print (set(uk_loc).intersection(set(us_locs)))
    return (us_locs)
    
def get_profile(path):
    profile = read_pickle1(path)
    #initially setting both UK and US to False
    profile['UK'] = False
    profile['US'] = False
    return (profile)



def geo_loc( path, uk_locs,us_locs):
    
    profile = get_profile(path)
    # set oouput directoty path
    outpath = '../data/location_data/UK_users/' + path.split('/')[-1]
    
    
    
    for i, row in profile.iterrows():
        for city in uk_locs:
            r1 = fuzz.token_set_ratio(city, row['location']) # looking for UK place in user 'location' 
            r2 = fuzz.token_set_ratio(city, row['description']) # looking for UK place in user 'description'

            # if match ratio>95, set UK to True
            if r1>95 or r2>95:
                profile.loc[i, 'UK'] = True 
                break

            

    for i, row in profile.iterrows():
        for city in us_locs+['America', 'United States', 'US']:
            r1 = fuzz.token_set_ratio(city, row['location']) # looking for US place in user 'location' 
            r2 = fuzz.token_set_ratio(city, row['description']) # looking for US place in user 'description'
            # if match ratio>95, set US to True
            if r1>95 or r2>95:
                profile.loc[i, 'US'] = True
                break
    

    


    for i, row in profile.iterrows(): 
        try:
            if 'london' in row['location'].lower(): # special case of London, setting to UK
                profile.loc[i, 'UK'] =True 
                profile.loc[i, 'US'] =False
        except:
            pass
        # if place exists in both US and UK, set UK to True only if place also contains ['England', 'United Kingdom', 'UK', 'Greater Manchester']
        if (profile.loc[i, 'US'] ==True and profile.loc[i, 'UK']== True):
            try:
                if any(s.lower() in row['location'].lower() for s in ['England', 'United Kingdom', 'UK', 'Greater Manchester']):
                    profile.loc[i, 'US'] =False
                else:
                    profile.loc[i, 'UK'] =False
            except:
                profile.loc[i, 'UK'] =False
                profile.loc[i, 'US'] =False
        

        #  set UK to True only if place does not contain ['America', 'United States', 'US', 'USA', 'NY', 'TX', 'CA', 'MI']
        if (profile.loc[i, 'UK']== True and profile.loc[i, 'US']== False):
            try:
                if any(s.lower() in row['location'].lower() for s in ['America', 'United States', 'US', 'USA', 'NY', 'TX', 'CA', 'MI', 'NC']):
                    profile.loc[i, 'UK'] =False
                if row['location'].lower()=='ireland' and row['location'].lower()!='northern ireland': # special case for northern ireland
                    profile.loc[i, 'UK'] =False 

            except:
                profile.loc[i, 'UK'] =False
                profile.loc[i, 'US'] =False

    #saving file          
    profile.to_csv(outpath.replace('.pkl','.csv'))
    profile.to_pickle(outpath)
    
    uk = profile[profile['UK']==True]['id'].tolist()
    us = profile[profile['US']==True]['id'].tolist()
    print ("intersection")
    print (len(set(uk).intersection(set(us))))
                


if __name__ == '__main__':
    
    uk_locs = get_uk('../data/location_data/ukloc.csv') # getting UK locations to be used
    us_locs = get_uk_us('../data/location_data/US_UK_common_places.csv', uk_locs) # getting location with same names in US and UK
    
    
    print (uk_locs)

    all_paths  = glob.glob('../data/user_profiles/all_user_profiles/*.pkl') # getting user profiles
    
    # utilizing multiple CPUs.
    n_p = os.cpu_count()-15 

    print ("utilizing number of cpus", n_p)
    # running for each file per thread
    pool = Pool(n_p)
    pool.starmap(geo_loc, zip(all_paths, [uk_locs]*len(all_paths),  [us_locs]*len(all_paths)))
    pool.close()
    
