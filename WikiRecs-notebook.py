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

# +
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
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from tqdm.auto import tqdm
import umap
import pickle
import collections
import recommenders
import plotly.express as px
from pyarrow import feather
import itertools
# -

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# # Get recent changes (to get a list of users)

# +
# df = wr.get_sample_of_users(edit_lookback=100000, outfile="sampled_users.pickle")

# +
# Check a user

wr.print_user_history(histories_train, user="Yoninah")

# +
# Change column types

all_histories.loc[:,'userid'] = all_histories.loc[:,'userid'].astype(int)

all_histories['title'] = all_histories['title'].astype(str)
# -

feather.write_feather(all_histories, '../all_histories.feather')

# # Load all_histories (raw data), transform and split 

# +
# %%time
all_histories = feather.read_feather('../all_histories.feather')

print("Length raw edit history data: {}".format(len(all_histories)))
# -

username

# +
## Add one particular user
cols = ['userid', 'user', 'pageid', 'title',
       'timestamp', 'sizediff']

with open("../username.txt", "r") as file:
    for username in file:
        oneuser = wr.get_edit_history(user=username.strip(),
                            latest_timestamp="2020-12-30T02:02:09Z",
                            earliest_timestamp="2019-12-30T02:02:09Z")
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
feather.write_feather(clean_histories, '../clean_histories.feather')

# ## Build lookup tables

# %%time
clean_histories = feather.read_feather('../clean_histories.feather')

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

wr.save_pickle((p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t), '../lookup_tables.pickle')

wr.save_pickle((userids, pageids), '../users_and_pages.pickle')

#
# ## Build test and training set

p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('../lookup_tables.pickle')
userids, pageids = wr.load_pickle('../users_and_pages.pickle')

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

print("Number of pages in training set: {}".format(len(histories_train.pageid.unique())))
print("Number of users in training set: {}".format(len(histories_train.userid.unique())))
print("Number of pages with > 1 user editing: {}".format(np.sum(histories_train.drop_duplicates(subset=['title','user']).groupby('title').count().user > 1)))

feather.write_feather(histories_train, '../histories_train.feather')
feather.write_feather(histories_dev, '../histories_dev.feather')
feather.write_feather(histories_test, '../histories_test.feather')

# +
resurface_userids, discovery_userids = wr.get_resurface_discovery(histories_train, histories_dev)

print("%d out of %d userids are resurfaced (%.1f%%)" % (len(resurface_userids), len(userids), 100*float(len(resurface_userids))/len(userids)))
print("%d out of %d userids are discovered (%.1f%%)" % (len(discovery_userids), len(userids), 100*float(len(discovery_userids))/len(userids)))
# -

wr.save_pickle((resurface_userids, discovery_userids), '../resurface_discovery_users.pickle')

# # Build matrix for implicit collaborative filtering

# +
# %%time

# Get the user/page edit counts
for_implicit = histories_train.groupby(["userid","pageid"]).count().first_timestamp.reset_index().rename(columns={'first_timestamp':'edits'})
for_implicit.loc[:,'edits'] = for_implicit.edits.astype(np.int32)
# -

wr.save_pickle(for_implicit, '../for_implicit.pickle')

# ## Test pivot that only works on a smaller set

# +
# for_implicit = wr.load_pickle('../for_implicit.pickle')
# for_implicit = for_implicit.pivot(index='pageid',columns='userid',values='edits').fillna(0).astype(int)

# pageid_to_index = {pageid:i for i, pageid in enumerate(pageids)}


# for_implicit = csc_matrix(for_implicit.values)

# model = implicit.als.AlternatingLeastSquares(factors=50)


# model.fit(for_implicit)


# p = pageid_to_index[561315]

# page_lookup.loc[pageids[p]]

# results = model.similar_items(p,20)

# page_lookup.loc[[pageids[r[0]] for r in results]]
# -

# ## Build large co-occurrence matrix incrementally

for_implicit = wr.load_pickle('../for_implicit.pickle')

# Make empty sparse matrix
implicit_matrix = lil_matrix((len(pageids), len(userids)), dtype=np.int32)
implicit_matrix.shape

# Fill in the matrix with the edit counts. iterrows is 10x slower!
for pageid, userid, edits  in tqdm(for_implicit.loc[:,['pageid','userid','edits']].values.tolist(), total=len(for_implicit)):
    implicit_matrix[p2i[pageid], u2i[userid]] = edits    

implicit_matrix = csc_matrix(implicit_matrix)

# %%time
wr.save_pickle(implicit_matrix,'../implicit_matrix.pickle')

# ### Test the matrix and indices

implicit_matrix = wr.load_pickle('../implicit_matrix.pickle')

indices = np.flatnonzero(implicit_matrix[:,n2i[username.strip()]].toarray())

# +
# Crude item to item recs by looking for items edited by the same editors (count how many editors overlap)

veditors = np.flatnonzero(implicit_matrix[t2i['Steven Universe'],:].toarray())

indices =  np.flatnonzero(np.sum(implicit_matrix[:,veditors] > 0,axis=1))

totals = np.asarray(np.sum(implicit_matrix[:,veditors] > 0 ,axis=1)[indices])

sorted_order = np.argsort(totals.squeeze())

[i2t.get(i, "")  + " " + str(total[0]) for i,total in zip(indices[sorted_order],totals[sorted_order])][::-1]
# -

# Histories of editors who had that item
for ved in veditors:
    print("\n\n\n" + i2n[ved])
    wr.print_user_history(all_histories, user=i2n[ved])

# +
# # Try a variation with tfidf
# from sklearn.feature_extraction.text import TfidfTransformer
# implicit_tfidf = TfidfTransformer().fit_transform(implicit_matrix)

# veditors = np.flatnonzero(implicit_tfidf[t2i['South Park'],:].toarray())

# indices =  np.flatnonzero(np.sum(implicit_tfidf[:,veditors] > 0,axis=1))

# totals = np.asarray(np.sum(implicit_tfidf[:,veditors],axis=1)[indices])

# sorted_order = np.argsort(totals.squeeze())
# [i2t[i]  + " " + str(total[0]) for i,total in zip(indices[sorted_order],totals[sorted_order])][::-1]
# -

# # Implicit recommendation

# +
num_factors = 100
regularization = 0.01

model = implicit.als.AlternatingLeastSquares(
    factors=num_factors, regularization=regularization
)
model.fit(implicit_matrix)
# -

wr.save_pickle(model,'als%d_model.pickle' % num_factors)

# # Evaluate models

# ## Item to item recommendation

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
# -

# Only plot the ones with over 3 entries
indices = np.squeeze(np.asarray(np.sum(implicit_matrix[nonzero,:],axis=1))) > 3

indices = nonzero[indices]

len(indices)

# Visualize  the collaborative filtering item vectors, embedding into 2D space with UMAP
# nonzero = np.flatnonzero(implicit_matrix.sum(axis=1))
# indices = nonzero[::100]
embedding = umap.UMAP().fit_transform(model.item_factors[indices[::100],:])

plt.figure(figsize=(10,10))
plt.plot(embedding[:,0], embedding[:,1],'.')
# _ = plt.axis('square')

# +
# Visualize actors in the embeddings space

edit_counts = np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1)))
log_edit_counts = np.log10(np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1))))

emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })

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

# +
# Load the edit histories in the training set and the test set
histories_train = feather.read_feather('../histories_train.feather')
histories_test = feather.read_feather('../histories_test.feather')

implicit_matrix = wr.load_pickle('../implicit_matrix.pickle')
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('../lookup_tables.pickle')

userids, pageids = wr.load_pickle('../users_and_pages.pickle')


# -

# # Evaluate on test set

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

results = {}

# %%time
K=20
prec = recommenders.PopularityRecommender(histories_train)
precs = prec.recommend_all(userids, K)

rec_name = "Popularity"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, precs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)


results[rec_name]

# +
# %%time
# Most recent
K=20
rrec = recommenders.MostRecentRecommender(histories_train)
rrecs = rrec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Recent"

results[rec_name] = wr.get_recs_metrics(
    histories_dev, rrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
# -

# %%time
# Sorted by frequency of edits
K=20
frec = recommenders.MostFrequentRecommender(histories_train)
frecs = frec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Frequent"

results[rec_name] = wr.get_recs_metrics(
    histories_dev, frecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

# ### Implicit collaborative filtering

wr.load_pickle('../implicit_matrix.pickle')

num_factors = 10

model = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=0.01)

# +
# implicit_matrix[implicit_matrix > 1] = 1
# -

# %%time
model.fit(implicit_matrix)


wr.save_pickle(model,'../als%d_model.pickle' % num_factors)

model = wr.load_pickle('../als100_model.pickle')

# %%time
K=20
irec = recommenders.ImplicitCollaborativeRecommender(model, implicit_matrix)
irecs = irec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p)
rec_name = "Implicit"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, irecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]

# %%time
# Sorted by Jaccard
K=20
recent_pages_dict = rrec.all_recent_only(K, userids,  interactions=histories_train)
jrec = recommenders.JaccardRecommender(implicit_matrix, p2i=p2i, t2i=t2i, i2t=i2t, i2p=i2p, n2i=n2i, u2i=u2i, i2u=i2u)
jrecs = jrec.recommend_all(userids, 
                                   K, 
                                   num_lookpage_pages=1, 
                                   recent_pages_dict=recent_pages_dict, 
                                   interactions=histories_train)

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
for title in tqdm([i2t[i] for i in user_page_indices], total=len(user_page_indices)):
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


# +
index = list(range(len(edited_pages)))
np.random.shuffle(index)

for i in index[:10]:
    print(edited_pages.index[i])
    wr.print_user_history(user=edited_pages.index[i],all_histories=clean_histories)
    print("\n\n\n")
# -

sns.distplot(edited_pages,kde=False,bins=np.arange(0,2000,20))

clean_histories.loc[clean_histories.user=="HyprMarc"].iloc[:20]

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


