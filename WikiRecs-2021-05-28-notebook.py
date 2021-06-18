# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: wikirecs
#     language: python
#     name: wikirecs
# ---

# # WikiRecs
# A project to recommend the next Wikipedia article you might like to edit

# + init_cell=true
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import wikipedia
import requests
import os
import wikirecs as wr
import implicit
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix
from tqdm.auto import tqdm
import umap
import pickle
import collections
import recommenders
import plotly.express as px
from pyarrow import feather
import itertools
from itables import show


from implicit.nearest_neighbours import (
    bm25_weight)
# -

from itables.javascript import load_datatables
load_datatables()

# + init_cell=true
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)

# + init_cell=true
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
# -

# # Assemble the complete histories

import os
all_histories = []
for fname in os.listdir('edit_histories_2021-05-28'):
    if 'feather' in  fname:
        all_histories.append(feather.read_feather('edit_histories_2021-05-28/{}'.format(fname)))

all_histories = pd.concat(all_histories, ignore_index=True)

feather.write_feather(all_histories, "all_histories_2021-05-28.feather")

# %%time
all_histories = feather.read_feather("all_histories_2021-05-28.feather")


all_histories.columns

len(all_histories.pageid.unique())

# # Load all_histories (raw data), transform and split 

# %%time
all_histories = feather.read_feather("all_histories_2021-05-28.feather")

print("Length raw edit history data: {}".format(len(all_histories)))

# +
# %%time
all_histories = feather.read_feather("all_histories_2021-05-28.feather")

print("Length raw edit history data: {}".format(len(all_histories)))

# +
from pull_edit_histories import get_edit_history

## Add one particular user
cols = ['userid', 'user', 'pageid', 'title',
       'timestamp', 'sizediff']

with open("../username.txt", "r") as file:
    for username in file:
        oneuser = get_edit_history(user=username.strip(),
                            latest_timestamp="2021-05-28T22:02:09Z",
                            earliest_timestamp="2020-05-28T22:02:09Z")
        oneuser = pd.DataFrame(oneuser).loc[:,cols]
        all_histories = pd.concat([all_histories, oneuser], ignore_index=True)

print("Length after adding users: {}".format(len(all_histories)))
# -

# ## EDA on raw histories

# Look at the distribution of edit counts
edit_counts = all_histories.groupby('userid').userid.count().values

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
sns.distplot(edit_counts,kde=False,bins=np.arange(0,20000,200))
plt.xlabel('Number of edits by user')
plt.subplot(1,2,2)
sns.distplot(edit_counts,kde=False,bins=np.arange(0,200,1))
plt.xlim([0,200])
plt.xlabel('Number of edits by user')
num_counts = len(edit_counts)
print("Median edit counts: %d" % np.median(edit_counts))
thres = 5
over_thres = np.sum(edit_counts > thres)
print("Number over threshold %d: %d (%.f%%)" % (thres, over_thres, 100*over_thres/num_counts))

# Most edits by user
all_histories.groupby(['userid','user']).userid.count().sort_values(ascending=False)

# Find the elbow in number of edits
plt.plot(all_histories.groupby(['userid','user']).userid.count().sort_values(ascending=False).values)
# plt.ylim([0,20000])

# +
# What are the most popular pages (edited by the most users)
page_popularity = all_histories.drop_duplicates(subset=['title','user']).groupby('title').count().user.sort_values()

pd.set_option('display.max_rows', 1000)
page_popularity.iloc[-1000:].iloc[::-1]


# -

# ## Clean data

# ### Remove consecutive edits and summarize runs

# +
# %%time
def remove_consecutive_edits(df):
    c = dict(zip(df.columns, range(len(df.columns))))
    
    keyfunc = lambda x: (x[c['userid']],x[c['pageid']])
    first_and_last = lambda run: [run[0][c['userid']],
                                run[0][c['user']],
                                run[0][c['pageid']],
                                run[0][c['title']],
                                run[-1][c['timestamp']],
                                run[0][c['timestamp']],
                                sum([abs(r[c['sizediff']]) for r in run]),
                                len(run)]
    d  = df.values.tolist()
    return pd.DataFrame([first_and_last(list(g)) for k,g in itertools.groupby(d, key=keyfunc)], 
                        columns=['userid', 'user', 'pageid', 'title', 'first_timestamp', 'last_timestamp','sum_sizediff','consecutive_edits'])
                        
clean_histories = remove_consecutive_edits(all_histories)
# -

# ### Remove top N most popular pages

# +
# Get the top most popular pages
TOPN = 20
popularpages = all_histories.drop_duplicates(subset=['title','pageid','userid']).groupby(['title','pageid']).count().user.sort_values()[-TOPN:]

before_count = len(all_histories)
# -

popularpages

# Remove those popular pages
popular_pageids = popularpages.index.get_level_values(level='pageid').values
is_popular_page_edit = clean_histories.pageid.isin(popular_pageids)
clean_histories = clean_histories.loc[~is_popular_page_edit].copy()
all_histories = None
after_count = len(clean_histories)
print("%d edits (%.1f%%) were in top %d popular pages. Length after removing: %d" % (np.sum(is_popular_page_edit), 
                                                                                     100* np.sum(is_popular_page_edit)/before_count,
                                                                                     TOPN,
                                                                                     after_count)
     )

print("Number of unique page ids: {}".format(len(clean_histories.pageid.unique())))

# ### Remove users with too many or too few edits

MIN_EDITS = 5
MAX_EDITS = 10000

# Get user edit counts
all_user_edit_counts = clean_histories.groupby(['userid','user']).userid.count()

# +
# Remove users with too few edits
keep_user = all_user_edit_counts.values >= MIN_EDITS

# Remove users with too many edits
keep_user = keep_user & (all_user_edit_counts.values <= MAX_EDITS)

# Remove users with "bot" in the name
is_bot = ['bot' in username.lower() for username in all_user_edit_counts.index.get_level_values(1).values]
keep_user = keep_user & ~np.array(is_bot)
print("Keep %d users out of %d (%.1f%%)" % (np.sum(keep_user), len(all_user_edit_counts), 100*float(np.sum(keep_user))/len(all_user_edit_counts)))

# +
# Remove those users
userids_to_keep = all_user_edit_counts.index.get_level_values(0).values[keep_user]

clean_histories = clean_histories.loc[clean_histories.userid.isin(userids_to_keep)]

clean_histories = clean_histories.reset_index(drop=True)
# -

print("Length after removing users: {}".format(len(clean_histories)))

# %%time
# Save cleaned histories
feather.write_feather(clean_histories, '../clean_histories_2021-05-28.feather')

# ## Build lookup tables

# %%time
clean_histories = feather.read_feather('../clean_histories_2021-05-28.feather')

# +
# Page id to title and back
lookup = clean_histories.drop_duplicates(subset=['pageid']).loc[:,['pageid','title']]
p2t = dict(zip(lookup.pageid, lookup.title))
t2p = dict(zip(lookup.title, lookup.pageid))

# User id to name and back
lookup = clean_histories.drop_duplicates(subset=['userid']).loc[:,['userid','user']]
u2n = dict(zip(lookup.userid, lookup.user))
n2u = dict(zip(lookup.user, lookup.userid))


# +
# Page id and userid to index in cooccurence matrix and back
pageids = np.sort(clean_histories.pageid.unique())
userids = np.sort(clean_histories.userid.unique())
 
p2i = {pageid:i for i, pageid in enumerate(pageids)}
u2i = {userid:i for i, userid in enumerate(userids)}


i2p = {v: k for k, v in p2i.items()}
i2u = {v: k for k, v in u2i.items()}


# +
# User name and page title to index and back
n2i = {k:u2i[v] for k, v in n2u.items() if v in u2i}
t2i = {k:p2i[v] for k, v in t2p.items() if v in p2i}

i2n = {v: k for k, v in n2i.items()}
i2t = {v: k for k, v in t2i.items()}
# -

wr.save_pickle((p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t), '../lookup_tables_2021-05-28.pickle')

wr.save_pickle((userids, pageids), '../users_and_pages_2021-05-28.pickle')

#
# ## Build test and training set

p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('../lookup_tables_2021-05-28.pickle')
userids, pageids = wr.load_pickle('../users_and_pages_2021-05-28.pickle')

# Make a test set from the most recent edit by each user
histories_test = clean_histories.groupby(['userid','user'],as_index=False).first()

# Subtract it from the rest to make the training set
histories_train = wr.dataframe_set_subtract(clean_histories, histories_test)
histories_train.reset_index(drop=True, inplace=True)

# Make a dev set from the second most recent edit by each user
histories_dev = histories_train.groupby(['userid','user'],as_index=False).first()
# Subtract it from the rest to make the final training set
histories_train = wr.dataframe_set_subtract(histories_train, histories_dev)
histories_train.reset_index(drop=True, inplace=True)

print("Length of test set: {}".format(len(histories_test)))
print("Length of dev set: {}".format(len(histories_dev)))
print("Length of training after removal of test: {}".format(len(histories_train)))

print("Number of pages in training set: {}".format(len(histories_train.pageid.unique())))
print("Number of users in training set: {}".format(len(histories_train.userid.unique())))
print("Number of pages with > 1 user editing: {}".format(np.sum(histories_train.drop_duplicates(subset=['title','user']).groupby('title').count().user > 1)))

feather.write_feather(histories_train, '../histories_train_2021-05-28.feather')
feather.write_feather(histories_dev, '../histories_dev_2021-05-28.feather')
feather.write_feather(histories_test, '../histories_test_2021-05-28.feather')

# +
resurface_userids, discovery_userids = wr.get_resurface_discovery(histories_train, histories_dev)

print("%d out of %d userids are resurfaced (%.1f%%)" % (len(resurface_userids), len(userids), 100*float(len(resurface_userids))/len(userids)))
print("%d out of %d userids are discovered (%.1f%%)" % (len(discovery_userids), len(userids), 100*float(len(discovery_userids))/len(userids)))
# -

wr.save_pickle((resurface_userids, discovery_userids), '../resurface_discovery_users_2021-05-28.pickle')

# # FIG Rama and other examples

print("Number of edits by Rama in a year: {}".format(len(all_histories.loc[all_histories.user == 'Rama'])))
print("Number of pages edited: {}".format(len(all_histories.loc[all_histories.user == 'Rama'].drop_duplicates(subset=['pageid']))))


# +
from pull_edit_histories import get_edit_history
oneuser = get_edit_history(user="Thornstrom",
                            latest_timestamp="2021-05-28T22:02:09Z",
                            earliest_timestamp="2020-05-28T22:02:09Z")

oneuser = pd.DataFrame(oneuser).loc[:,cols]

# -

wr.print_user_history(all_histories, user="Rama")



wr.print_user_history(all_histories, user="Meow")

# # Build matrix for implicit collaborative filtering

# +
# %%time

# Get the user/page edit counts
for_implicit = histories_train.groupby(["userid","pageid"]).count().first_timestamp.reset_index().rename(columns={'first_timestamp':'edits'})
for_implicit.loc[:,'edits'] = for_implicit.edits.astype(np.int32)

# +
row = np.array([p2i[p] for p in for_implicit.pageid.values])
col = np.array([u2i[u] for u in for_implicit.userid.values])

implicit_matrix_coo = coo_matrix((for_implicit.edits.values, (row, col)))


implicit_matrix = csc_matrix(implicit_matrix_coo)
# -

# %%time
wr.save_pickle(implicit_matrix,'../implicit_matrix_2021-05-28.pickle')

# ### Test the matrix and indices

implicit_matrix = wr.load_pickle('../implicit_matrix_2021-05-28.pickle')

indices = np.flatnonzero(implicit_matrix[:,n2i[username.strip()]].toarray())

# +
# Crude item to item recs by looking for items edited by the same editors (count how many editors overlap)

veditors = np.flatnonzero(implicit_matrix[t2i['Hamburger'],:].toarray())

indices =  np.flatnonzero(np.sum(implicit_matrix[:,veditors] > 0,axis=1))

totals = np.asarray(np.sum(implicit_matrix[:,veditors] > 0 ,axis=1)[indices])

sorted_order = np.argsort(totals.squeeze())

[i2t.get(i, "")  + " " + str(total[0]) for i,total in zip(indices[sorted_order],totals[sorted_order])][::-1]
# -

# Histories of editors who had that item
for ved in veditors:
    print("\n\n\n" + i2n[ved])
    wr.print_user_history(all_histories, user=i2n[ved])

# # Implicit recommendation

implicit_matrix = wr.load_pickle('../implicit_matrix_2021-05-28.pickle')
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('../lookup_tables_2021-05-28.pickle')

bm25_matrix = bm25_weight(implicit_matrix, K1=100, B=0.25)

num_factors =200
regularization = 0.01
os.environ["OPENBLAS_NUM_THREADS"] = "1"
model = implicit.als.AlternatingLeastSquares(
    factors=num_factors, regularization=regularization
)
model.fit(bm25_matrix)

wr.save_pickle(model,'../als%d_bm25_model.pickle' % num_factors)

model = wr.load_pickle('../als200_bm25_model_2021-05-28.pickle')

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Steven Universe'],1000)
['%s %.4f %d' % (i2t[ind], score, (implicit_matrix[ind,:]>0).sum()) for ind, score in results if ((implicit_matrix[ind,:]>0).sum() > 4)]

results = model.similar_items(t2i['George Clooney'],20)
['%s %.4f %d' % (i2t[ind], score, (implicit_matrix[ind,:]>0).sum()) for ind, score in results]

results = model.similar_items(t2i['Hamburger'],20)
['%s %.4f %d' % (i2t[ind], score, (implicit_matrix[ind,:]>0).sum()) for ind, score in results]

u = n2u["Rama"]
recommendations = model.recommend(u2i[u], bm25_matrix.tocsc(), N=1000, filter_already_liked_items=False)
[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) + ' %d' % (implicit_matrix[ind,:]>0).sum()
 for ind, score in recommendations]

# ## Grid search results

grid_search_results = wr.load_pickle("../implicit_grid_search.pickle")

pd.DataFrame(grid_search_results)


pd.DataFrame([[i['num_factors'], i['regularization']] + list(i['metrics'].values()) for i in grid_search_results],
            columns = ['num_factors','regularization'] + list(grid_search_results[0]['metrics'].keys()))


grid_search_results_bm25 = wr.load_pickle("../implicit_grid_search_bm25.pickle")

pd.DataFrame([[i['num_factors'], i['regularization']] + list(i['metrics'].values()) for i in grid_search_results_bm25],
            columns = ['num_factors','regularization'] + list(grid_search_results_bm25[0]['metrics'].keys()))


# # B25 Recommendation

from implicit.nearest_neighbours import BM25Recommender

# +
bm25_matrix = bm25_weight(implicit_matrix, K1=20, B=1)
bm25_matrix = bm25_matrix.tocsc()
sns.distplot(implicit_matrix[implicit_matrix.nonzero()],bins = np.arange(0,100,1),kde=False)

sns.distplot(bm25_matrix[bm25_matrix.nonzero()],bins = np.arange(0,100,1),kde=False)
# -

implicit_matrix.shape

K1 = 100
B = 0.25
model = BM25Recommender(K1, B)
model.fit(implicit_matrix)

wr.save_pickle(model, '../bm25_model_2021-05-28.pkl')

results = model.similar_items(t2i['Mark Hamill'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['George Clooney'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['George Clooney'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Lil Nas X'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Lil Nas X'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Hamburger'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Hamburger'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

# +
# model.recommend?
# -

type(implicit_matrix)

implicit_matrix.astype(np.int16)

implicit_matrix.dtype

from implicit._nearest_neighbours import NearestNeighboursScorer

n  = NearestNeighboursScorer

# +
# n.recommend?
# -

u = n2u["Rama"]
recommendations = model.recommend(u2i[u], implicit_matrix.astype(np.float32), N=1000, filter_already_liked_items=True)

[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) 
 for ind, score in recommendations]

plt.plot([ score for i,(ind, score) in enumerate(recommendations) if implicit_matrix[ind,u2i[u]]==0])

wr.save_pickle(model, "b25_model.pickle")

model = wr.load_pickle("b25_model.pickle")

grid_search_results = wr.load_pickle("../bm25_grid_search.pickle")

grid_search_results

grid_search_results[32]

pd.DataFrame([[i['K1'], i['B']] + list(i['metrics'].values()) for i in grid_search_results],
            columns = ['K1','B'] + list(grid_search_results[0]['metrics'].keys()))


# # Evaluate models

# ## Item to item recommendation

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['George Clooney'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

results = model.similar_items(t2i['Michael Jackson'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]

# +
# np.dot(np.reshape(model.user_factors[n2i['DoctorWho42']], (-1, 1)).T,
# np.reshape(model.item_factors[1379170], (-1, 1)))
# -

['%s %.4f' % (i2t[ind], score) for ind, score in results]

# ## User to item recommendations

# +
# Check out a specific example

u = n2u["HyprMarc"]

wr.print_user_history(clean_histories, userid=u)

# +
# Check out a specific example

u = n2u["HyprMarc"]

wr.print_user_history(clean_histories, userid=u)
# -

u = n2u["HyprMarc"]
recommendations = model.recommend(u2i[u], implicit_matrix, N=100, filter_already_liked_items=False)

[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) 
 for ind, score in recommendations]

# # Visualize implicit embeddings

model = wr.load_pickle('../als150_model.pickle')

# +
# Visualize the collaborative filtering user vectors
nonzero = np.flatnonzero(implicit_matrix.sum(axis=0))
embedding = umap.UMAP().fit_transform(model.user_factors[nonzero])

plt.figure(figsize=(10,10))
plt.plot(embedding[:,0], embedding[:,1],'.')
# _ = plt.axis('square')

# +
# Only plot the ones with over 3 entries
indices = np.squeeze(np.asarray(np.sum(implicit_matrix[nonzero,:],axis=1))) > 3

indices = nonzero[indices]
# -

len(indices)

# Visualize  the collaborative filtering item vectors, embedding into 2D space with UMAP
# nonzero = np.flatnonzero(implicit_matrix.sum(axis=1))
# indices = nonzero[::100]
embedding = umap.UMAP().fit_transform(model.item_factors[indices,:])

plt.figure(figsize=(10,10))
plt.plot(embedding[:,0], embedding[:,1],'.')
# _ = plt.axis('square')

# ## Visualize actors in the embeddings space

# +
edit_counts = np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1)))
log_edit_counts = np.log10(np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1))))

emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })
# -

actors = ['Mark Hamill',
'Carrie Fisher',
'James Earl Jones',
'David Prowse',
'Sebastian Shaw (actor)',
'Alec Guinness',
'Jake Lloyd',
'Hayden Christensen',
'Ewan McGregor',
'William Shatner',
'Leonard Nimoy',
'DeForest Kelley',
'James Doohan',
'George Takei']
actor_indices = [t2i[a] for a in actors]
edit_counts = np.squeeze(np.asarray(np.sum(implicit_matrix[actor_indices,:],axis=1)))
log_edit_counts = np.log10(np.squeeze(np.asarray(np.sum(implicit_matrix[actor_indices,:],axis=1))))
embedding = umap.UMAP().fit_transform(model.item_factors[actor_indices,:])
emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in actor_indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })
key = np.zeros(len(actors))
key[:8] = 1
fig = px.scatter(data_frame=emb_df,
                 x='dim1',
                 y='dim2', 
                 hover_name='title',
                 color=key,
                 hover_data=['edit_count'])
fig.update_layout(
    autosize=False,
    width=600,
    height=600,)
fig.show()

# +
# Full embedding plotly interactive visualization

emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })

fig = px.scatter(data_frame=emb_df,
                 x='dim1',
                 y='dim2', 
                 hover_name='title',
                 color='log_edit_count',
                 hover_data=['edit_count'])
fig.update_layout(
    autosize=False,
    width=600,
    height=600,)
fig.show()
# -

# # Evaluate on test set

# +
# Load the edit histories in the training set and the test set
histories_train = feather.read_feather('../histories_train_2021-05-28.feather')
histories_test = feather.read_feather('../histories_test_2021-05-28.feather')
histories_dev = feather.read_feather('../histories_dev_2021-05-28.feather')

implicit_matrix = wr.load_pickle('../implicit_matrix_2021-05-28.pickle')
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('../lookup_tables_2021-05-28.pickle')

userids, pageids = wr.load_pickle('../users_and_pages_2021-05-28.pickle')


# -

resurface_userids, discovery_userids   = wr.load_pickle('../resurface_discovery_users_2021-05-28.pickle')

results = {}

wr.display_recs_with_history(
    recs,
    userids[:100],
    histories_test,
    histories_train,
    p2t,
    u2n,
    recs_to_display=5,
    hist_to_display=10,
)

# ## Most popular

# +
# %%time
K=20
rec_name = "Popularity"

prec = recommenders.PopularityRecommender(histories_train)
precs = prec.recommend_all(userids, K)
wr.save_pickle(precs, "../" + rec_name +"_recs.pickle")


# +

results[rec_name] = wr.get_recs_metrics(
    histories_dev, precs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)

results[rec_name]
# -


# ## Most recent

# %%time
# Most recent
K=20
rrec = recommenders.MostRecentRecommender(histories_train)
rrecs = rrec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Recent"
wr.save_pickle(rrecs, "../" + rec_name +"_recs.pickle")

len(resurface_userids)

results ={}

results[rec_name] = wr.get_recs_metrics(
    histories_dev, rrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

# ## Most frequent

# %%time
# Sorted by frequency of edits
K=20
frec = recommenders.MostFrequentRecommender(histories_train)
frecs = frec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Frequent"
wr.save_pickle(frecs, "../" + rec_name +"_recs.pickle")


results[rec_name] = wr.get_recs_metrics(
    histories_dev, frecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

# ## BM25

# %%time
K=20
brec = recommenders.MyBM25Recommender(model, implicit_matrix)

brecs = brec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p, filter_already_liked_items=False)
rec_name = "bm25"
wr.save_pickle(brecs, "../" + rec_name +"_recs.pickle")

# filter_already_liked_items = False
results[rec_name] = wr.get_recs_metrics(
    histories_dev, brecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

# filter_already_liked_items = True
rec_name = "bm25_filtered"
brecs_filtered = brec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p, filter_already_liked_items=True)
wr.save_pickle(brecs_filtered, "../" + rec_name +"_recs.pickle")


results[rec_name] = wr.get_recs_metrics(
    histories_dev, brecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

# ## Implicit collaborative filtering

model_als = wr.load_pickle('../als200_bm25_model_2021-05-28.pickle')

model = None

# %%time
rec_name = "als"
K=20
irec = recommenders.ImplicitCollaborativeRecommender(model_als, bm25_matrix.tocsc())
irecs = irec.recommend_all(userids, K, i2p=i2p, filter_already_liked_items=False)
wr.save_pickle(irecs, "../" + rec_name +"_recs.pickle")

results[rec_name] = wr.get_recs_metrics(
    histories_dev, irecs, K, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)
results[rec_name]

# Old recs
rec_name = "als_filtered"
wr.get_recs_metrics(
    histories_dev, irecs, K, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)





rec_name = "als_filtered"
K=20
irec = recommenders.ImplicitCollaborativeRecommender(model_als, bm25_matrix.tocsc())
irecs_filtered = irec.recommend_all(userids, K, i2p=i2p, filter_already_liked_items=True)
results[rec_name] = wr.get_recs_metrics(
    histories_dev, irecs_filtered, K, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)
results[rec_name]

wr.save_pickle(irecs_filtered, "../" + rec_name +"_recs.pickle")

show(pd.DataFrame(results).T)

# ## Jaccard

# %%time
# Sorted by Jaccard
K=20
rrec = recommenders.MostRecentRecommender(histories_train)
recent_pages_dict = rrec.all_recent_only(K, userids,  interactions=histories_train)
jrec = recommenders.JaccardRecommender(implicit_matrix, p2i=p2i, t2i=t2i, i2t=i2t, i2p=i2p, n2i=n2i, u2i=u2i, i2u=i2u)
jrecs = jrec.recommend_all(userids, 
                                   K, 
                                   num_lookpage_pages=1, 
                                   recent_pages_dict=recent_pages_dict, 
                                   interactions=histories_train)

wr.save_pickle(jrecs,"jaccard-1_recs.pickle")

rec_name = "Jaccard"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, jrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

rec_name = "Jaccard"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, jrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

rec_name = "Jaccard"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, jrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

wr.display_recs_with_history(
    jrecs,
    userids[:30],
    histories_test,
    histories_train,
    p2t,
    u2n,
    recs_to_display=5,
    hist_to_display=10,
)

# %%time
# Sorted by Jaccard
K=5
jrec = recommenders.JaccardRecommender(implicit_matrix, p2i=p2i, t2i=t2i, i2t=i2t, i2p=i2p, n2i=n2i, u2i=u2i, i2u=i2u)
jrecs = jrec.recommend_all(userids[:1000], 
                                   10, 
                                   num_lookpage_pages=50, 
                                   recent_pages_dict=recent_pages_dict, 
                                   interactions=histories_train)
print("Jaccard")

K=5
print("Recall @ %d: %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K)))
print("Prop resurfaced: %.2f%%" % (100*wr.prop_resurface(jrecs, K, implicit_matrix, i2p, u2i)))
print("Recall @ %d (discovery): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=discovery_userids)))
print("Recall @ %d (resurface): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=resurface_userids)))

K=5
print("Recall @ %d: %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K)))
print("Prop resurfaced: %.2f%%" % (100*wr.prop_resurface(jrecs, K, implicit_matrix, i2p, u2i)))
print("Recall @ %d (discovery): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=discovery_userids)))
print("Recall @ %d (resurface): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=resurface_userids)))

K=5
print("Recall @ %d: %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K)))
print("Prop resurfaced: %.2f%%" % (100*wr.prop_resurface(jrecs, K, implicit_matrix, i2p, u2i)))
print("Recall @ %d (discovery): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=discovery_userids)))
print("Recall @ %d (resurface): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=resurface_userids)))

K=5
print("Recall @ %d: %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K)))
print("Prop resurfaced: %.2f%%" % (100*wr.prop_resurface(jrecs, K, implicit_matrix, i2p, u2i)))
print("Recall @ %d (discovery): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=discovery_userids)))
print("Recall @ %d (resurface): %.2f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=resurface_userids)))

print("Recall @ %d: %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K)))
print("Prop resurfaced: %.1f%%" % (100*wr.prop_resurface(jrecs, K, implicit_matrix, i2p, u2i)))
print("Recall @ %d (discovery): %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=discovery_userids)))
print("Recall @ %d (resurface): %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=resurface_userids)))

# ## Interleaved

recs.keys()

# +
# Interleaved jaccard and recent
K=20
rec_name = "Interleaved"
print(rec_name)
intrec = recommenders.InterleaveRecommender()
intrecs = intrec.recommend_all(K, [recs['Recent'], recs['bm25_filtered']])

wr.save_pickle(intrecs, "../" + rec_name +"_recs.pickle")
# -

results[rec_name] = wr.get_recs_metrics(
    histories_dev, intrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

results[rec_name] = wr.get_recs_metrics(
    histories_dev, intrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

results[rec_name] = wr.get_recs_metrics(
    histories_dev, intrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

# # Report on evaluations results

# ## Hard coded metrics

results = {}
results["Popularity"] = {'recall': 0.16187274312040842,
 'ndcg': 0.0005356797596941751,
 'resurfaced': 0.6213422985929523,
 'recall_discover': 0.11947959996459864,
 'recall_resurface': 0.2624396388830569,
 'ndcg_discover': 0.000410354483750028,
 'ndcg_resurface': 0.0008329819416998272}
results["Recent"] = {'recall': 22.618602913709378,
 'ndcg': 0.14306080818547054,
 'resurfaced': 71.13808990163118,
 'recall_discover': 0.03982653332153288,
 'recall_resurface': 76.18097837497375,
 'ndcg_discover': 0.00011494775493754298,
 'ndcg_resurface': 0.4821633227780786}
results["Frequent"] = {'recall': 20.834889802017184,
 'ndcg': 0.11356953338215306,
 'resurfaced': 76.10353629684971,
 'recall_discover': 0.035401362952473675,
 'recall_resurface': 70.17635943732941,
 'ndcg_discover': 9.90570471847343e-05,
 'ndcg_resurface': 0.38274923359395385}
results["Implicit"] = {'recall': 5.488108579255385,
 'ndcg': 0.026193145556306998,
 'resurfaced': 16.251556468683848,
 'recall_discover': 1.146119125586335,
 'recall_resurface': 15.788368675204703,
 'ndcg_discover': 0.004817135435898367,
 'ndcg_resurface': 0.0769022655123215}
results["Implicit_filtered"] = {'recall': 0.9027518366330469,
 'ndcg': 0.003856703716094881,
 'resurfaced': 0.0,
 'recall_discover': 1.2832994070271706,
 'recall_resurface': 0.0,
 'ndcg_discover': 0.005482465270193466,
 'ndcg_resurface': 0.0}
results["bm25"] = {'recall': 18.945336819823186,
 'ndcg': 0.1015175508656068,
 'resurfaced': 74.0469742248786,
 'recall_discover': 1.3939286662536507,
 'recall_resurface': 60.581566239764854,
 'ndcg_discover': 0.004204510293040833,
 'ndcg_resurface': 0.332367864833573}
results["bm25_filtered"] = {'recall': 1.0993992972911708,
 'ndcg': 0.004864465753718907,
 'resurfaced': 0.21421285277116625,
 'recall_discover': 1.7235188509874326,
 'recall_resurface': 0.03074085459575776,
 'ndcg_discover': 0.007525885340226556,
 'ndcg_resurface': 0.0003074085459575776}
results["interleaved"] = {'recall': 21.382766778732414,
 'ndcg': 0.12924273396038563,
 'resurfaced': 42.478676379031256,
 'recall_discover': 1.8364457031595716,
 'recall_resurface': 67.75141717404996,
 'ndcg_discover': 0.006943981897312752,
 'ndcg_resurface': 0.4193652616867473}


# ## Table of results

# +
results_df = pd.DataFrame(results).T

results_df.reset_index(inplace=True)
# -

results_df

results_df.sort_values("ndcg")

fig = px.scatter(data_frame=results_df,
                 x='ndcg_discover',
                 y='ndcg_resurface',
                hover_name='index')
#                  hover_name='title',)
fig.show()

fig = px.scatter(data_frame=results_df,
                 x='recall_discover',
                 y='recall_resurface',
                hover_name='index')
#                  hover_name='title',)
fig.show()

# ## Read recs in from files

recommender_names = ['Popularity', 'Recent', 'Frequent', 'Implicit', 'Implicit_filtered', 'bm25', 'bm25_filtered', 'interleaved']

recs = {rname:wr.load_pickle("../" + rname + "_recs.pickle") for rname in recommender_names}

# ## FIG Recall curves

histories_dev = feather.read_feather('../histories_dev_2021-05-28.feather')

plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20)
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)

plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20, discovery_userids)
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)

plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20, resurface_userids)
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)

# # User recommendation comparison

recommender_names


recs_subset = ["Recent","Frequent","Popularity","Implicit","Implicit_filtered","bm25","bm25_filtered"]

N=10





# +
def bold_viewed(val, viewed_pages):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    weight = 'bold' if val in  viewed_pages else 'normal'
    return 'font-weight: %s' % weight

def color_target(val, target_page):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val ==  target_page else 'black'
    return 'color: %s' % color



# -

print("Next edit: " + histories_dev.loc[histories_dev.userid == userid].title.values[0])


s.st


def display_user_recs_comparison(user_name, recs, recs_subset, train_set, test_set, N=20):
    userid = n2u[user_name]
    recs_table = pd.DataFrame({rec_name: [p2t[r] for r in recs[rec_name][userid][:N]] for rec_name in recs_subset})
    viewed_pages = train_set.loc[train_set.userid == userid,["title"]].drop_duplicates(subset=["title"]).values.squeeze()
    target_page = test_set.loc[test_set.userid == userid].title.values[0]
    print("Next edit: " + target_page)
    s = recs_table.style.applymap(bold_viewed, viewed_pages=viewed_pages).applymap(color_target, target_page=target_page)
    display(s)


display_user_recs_comparison('Thornstrom', recs, recs_subset, histories_train, histories_dev, N=10)

display(pd.DataFrame({rec_name: [p2t[r] for r in recs[rec_name][n2u['Rama']]][:N] for rec_name in recs_subset}))


histories_dev.loc[histories_dev.user == 'Yeeetttttboiiiii']

N = 20
display(pd.DataFrame({rec_name: [p2t[r] for r in recs[rec_name][n2u['HenryXVII']]][:N] for rec_name in recs_subset}))

persons_of_interest = [
    "DoctorWho42",
    "AxelSjögren",
    "Mighty platypus",
    "Tulietto",
    "LipaCityPH",
    "Hesperian Nguyen",
    "Thornstrom",
    "Meow",
    "HyprMarc",
    "Jampilot",
    "Rama"
]
N=10

irec_500 = recommenders.ImplicitCollaborativeRecommender(model, implicit_matrix)
irecs_poi = irec_500.recommend_all([n2u[user_name] for user_name in persons_of_interest], N, u2i=u2i, n2i=n2i, i2p=i2p)


# +

recs_list = [rrecs, frecs,  precs, jrecs, irecs_poi, brecs]
recs_names = ['Recent','Frequent','Popular','Jaccard similarity', 'Implicit collaborative', "BM25"]
pd.DataFrame({name: [p2t[r] for r in recs[n2u['Rama']]][:N] for recs, name in zip(recs_list, recs_names)})

for user_name in persons_of_interest:
    print(user_name)
    display(pd.DataFrame({name: [p2t[r] for r in recs[n2u[user_name]]][:N] for recs, name in zip(recs_list, recs_names)}))
    
# -

# # Jaccard recommender development

implicit_matrix = wr.load_pickle('../implicit_matrix.pickle')
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('../lookup_tables.pickle')

jrec = recommenders.JaccardRecommender(implicit_matrix, p2i=p2i, t2i=t2i, i2t=i2t, i2p=i2p, n2i=n2i, u2i=u2i, i2u=i2u)

# %%time
for title in 100*['Batman','Titanic','An American Tail','Batman Begins','Batman Forever']:
    jrec.item_to_item(N=10, title=title)

# +
user_name = 'DoctorWho42'

user_page_indices = np.flatnonzero(implicit_matrix[:,n2i[user_name]].A)
# -

# %%time
for title in tqdm([i2t[i] for i in user_page_indices], total=len(user_page_indices), leave=True):
    jrec.item_to_item(N=10, title=title)

my_indices = [i2t[i] for i in user_page_indices]

# %%time
d = jrec.jaccard_multiple(user_page_indices[:100], exclude_index=n2i[user_name])

# +
# d[d == 1] = np.nan

# d[d == 1] = 0

# +
mean_jaccard = np.nanmean(d,axis=1).A.squeeze()

order = np.argsort(mean_jaccard)[::-1]
# -

# %%time
jrecs = jrec.recommend(N=10, 
                       user=user_name, 
                       num_lookpage_pages=50, 
                       recent_pages_dict=rrecs, 
                       interactions=histories_train)

wr.prop_resurface(jrecs, K=10, implicit_matrix=implicit_matrix, i2p=i2p, u2i=u2i)

# %%time
user_name = "DoctorWho42"
recent_pages = rrec.recommend(N=50,user=user_name, interactions=histories_train)
user_page_indices = [p2i[p] for p in recent_pages]
d = jrec.jaccard_multiple(user_page_indices, exclude_index=n2i[user_name])

recs = jrec.recommend(N=10,user='DoctorWho42', num_lookpage_pages=50, recent_pages_dict=recent_pages_dict, interactions=histories_train)

[p2t[r] for r in recs]

# +
f = np.array(d)

# f = np.nan_to_num(f)

f[f==1] = np.nan
f[f==0] = np.nan


# +
m = np.nanmean(f,axis=0)
s = np.nanstd(f,axis=0)

f = f-m

f = f/s
# -

mean_jaccard = np.nanmax(f,axis=1).squeeze()
mean_jaccard = np.nan_to_num(mean_jaccard)
order = np.argsort(mean_jaccard)[::-1]

[p2t[p] for p in recent_pages[np.argmax(np.nan_to_num(f),axis=1)[order][:50]]]

[(i2t[o], mean_jaccard[o]) for o in order[:50] if o in i2t]

order = np.argsort(f[:,j])[::-1]

wr.print_user_history(histories_train, user='Da Vynci')

order

for j in range(50):
    p = recent_pages[j]
    print(p2t[p])
    order = np.argsort(f[:,j])[::-1]
    for o in order[:10]:
        print("   {} ({})".format(i2t[o],f[o,j]))

np.argsort(d,axis=0)

[(i2t[o], mean_jaccard[o]) for o in order if o in i2t]

[(i2t[o], mean_jaccard[o]) for o in order if o in i2t]

px.line(y=mean_jaccard[order][:5000], hover_name=[i2t[o] for o in order[:5000]])

[(i2t[o], mean_jaccard[o]) for o in order if o in i2t]

jrec.item_to_item(10, 'Leonard Nimoy')

type(implicit_matrix)

implicit_matrix.shape

plt.plot(implicit_matrix[order, n2i['DoctorWho42']].A.squeeze()[:250])

jrec.item_to_item(30,'Elton John')

plt.plot(mean_jaccard[order][:250])

plt.plot(mean_jaccard[order][127:250])

[(i2t[o], mean_jaccard[o]) for o in order if o in i2t]

user_page_indices

implicit_matrix_toy = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 0]])
assert implicit_matrix_toy.shape == (3, 4)
implicit_matrix_toy = csc_matrix(implicit_matrix_toy)
n2it = {"huey": 0, "dewey": 1, "louie": 2, "chewy": 3}
t2it = {"Batman": 0, "Mystery Men": 1, "Taxi Driver": 2}
i2nt = {v: k for k, v in n2it.items()}
i2tt = {v: k for k, v in t2it.items()}

jrec_toy = JaccardRecommender(implicit_matrix_toy, p2i=None, t2i=t2it, i2t=i2tt, i2p=None)


d = jrec_toy.jaccard_multiple([0,1])

mean_jaccard = np.mean(d,axis=1).A.squeeze()
print(mean_jaccard)
order = np.argsort(mean_jaccard)[::-1]
print(order)

d2 = d.copy()
d2[d2==1] = 0
print(d2)
mean_jaccard = np.nanmean(d2,axis=1).A.squeeze()
print(mean_jaccard)
order = np.argsort(mean_jaccard)[::-1]
print(order)

d2 = d.copy()
d2[d2==1] = np.nan
print(d2)
mean_jaccard = np.nanmean(d2,axis=1).A.squeeze()
print(mean_jaccard)
order = np.argsort(mean_jaccard)[::-1]
print(order)

# # Find interesting users

# +
edited_pages = clean_histories.drop_duplicates(subset=['title','user']).groupby('user').userid.count()

edited_pages = edited_pages[edited_pages > 50]
edited_pages = edited_pages[edited_pages < 300]
# -


clean_histories.columns

display_user_recs_comparison("Rama", recs, recs_subset, histories_train, histories_dev, N=20)


# +
index = list(range(len(edited_pages)))
np.random.shuffle(index)

for i in index[:10]:
    user_name = edited_pages.index[i]
    print(user_name)
    display_user_recs_comparison(user_name, recs, recs_subset, histories_train, histories_dev, N=20)
    print("\n\n\n")

# +
index = list(range(len(edited_pages)))
np.random.shuffle(index)

for i in index[:10]:
    print(edited_pages.index[i])
    display_user_recs_comparison
    wr.print_user_history(user=edited_pages.index[i],all_histories=clean_histories)
    print("\n\n\n")
# -

sns.distplot(edited_pages,kde=False,bins=np.arange(0,2000,20))

# # Repetition analysis

import itertools

clean_histories.head()

clean_histories.iloc[:1000].values.tolist()

x

clean_histories.columns

df = clean_histories
dict(zip(df.columns, range(len(df.columns))))

sum([1,2,3])

clean_histories.columns

identify_runs(clean_histories.iloc[:1000])


def identify_runs(df):
    d  = df.loc[:,['userid','pageid']].values.tolist()
    return [(k, len(list(g))) for k,g in itertools.groupby(d)]




# %%time
runs = identify_runs(clean_histories)

# +
lens = np.array([r[1] for r in runs])

single_edits = np.sum(lens==1)
total_edits = len(clean_histories)

print("Percent of edits that are part of a run: %.1f%%" % (100*(1-(float(single_edits)/total_edits))))

print("Percent of edits that are repetitions: %.1f%%" % (100*(1-len(runs)/total_edits)))
# -


