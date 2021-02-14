import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import wikipedia
import pickle 
import requests
import time
import wikirecs as wr




if __name__ == "__main__":
    histories = []
    cols = ['userid', 'user', 'pageid', 'title',
       'timestamp', 'sizediff']
    sampled_users = wr.load_pickle('sampled_users.pickle')
    start_time = time.time()
    print(len(sampled_users))
    print(sampled_users)
    
    sampled_users = sampled_users.reset_index()
    for i, (a, user,userid) in sampled_users.iterrows():
        if i < 5001:
            continue
        if np.mod(i,400) == 0:
            with wr.Timer("Saving to pickle"):
                wr.save_pickle(histories, 'edit_histories{}.pickle'.format(9000000+i))
            histories = []

        # with Timer("Pull the data"):
        thehistory = get_edit_history(int(userid))
        if len(thehistory) == 0:
            continue
            
        # with Timer("Convert to dataframe"):
        thehistory = pd.DataFrame(thehistory)
        # with Timer("Remove 'using'"):

        try:
            thehistory = thehistory[np.invert(thehistory.comment.astype(str).str.contains('using'))]
        except AttributeError:
            continue

        if len(thehistory) == 0:
            continue
        
        histories.append(thehistory.loc[:,cols])
        print("{}/{} {} ({}) has {} edits (time={})".format(i, len(sampled_users), user, int(userid), len(thehistory),time.time()-start_time))


with wr.Timer("Saving to pickle"):
    wr.save_pickle(histories, 'edit_histories{}.pickle'.format(i))

