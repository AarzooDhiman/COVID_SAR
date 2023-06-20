import pandas as pd
import pickle5 as pickle 

def read_pickle1(pth):
    with open(pth, "rb") as fh:
        data = pickle.load(fh)
    return (data)


df = read_pickle1('/disks/sdb/adhiman/SAR_data/data/user_timelines/276945706.pkl')

print (df.head())