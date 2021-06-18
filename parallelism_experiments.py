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

import dask.array as da
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# %%time
y = x + x.T
z = y[::2, 5000:].mean(axis=1)
z

# %%time
z.compute()

x = np.random.random((10000, 10000))

from math import sqrt
from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))

# %%time 
a = [recommender.recommend(userid=u, N=100, interactions=histories_train) for u in userids]

from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))

# %%time 
# from math import sqrt
# from joblib import Parallel, delayed
# a = Parallel(n_jobs=-1)(delayed(recommender.recommend)(u, 100, None, histories_train) for u in userids)

# # Do one the non-parallel way

p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('../lookup_tables.pickle')

userids, pageids = wr.load_pickle('../users_and_pages.pickle')

implicit_matrix = wr.load_pickle('../implicit_matrix.pickle')

mode l= wr.load_pickle('../als64_bm25_model.pickle')

bm25_matrix = bm25_weight(implicit_matrix, K1=100, B=0.25)

num_factors =200
regularization = 0.01
os.environ["OPENBLAS_NUM_THREADS"] = "1"
model = implicit.als.AlternatingLeastSquares(
    factors=num_factors, regularization=regularization
)
model.fit(bm25_matrix)

K=20
irec = recommenders.ImplicitCollaborativeRecommender(model, bm25_matrix.tocsc())
irecs = irec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p)
rec_name = "Implicit"

K=20
irec = recommenders.ImplicitCollaborativeRecommender(model, bm25_matrix.tocsc())
irecs = irec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p)
rec_name = "Implicit"

recs = model.recommend_all(bm25_matrix.tocsc().T, K, filter_already_liked_items=False)

irecs

userids

recs

recs = model.recommend_all(bm25_matrix.tocsc().T, K*5)

recs.shape

irecs[n2u[name]]

# +
name = 'Rama'

recommendations = irecs[n2u[name]]

[ ("*" if implicit_matrix[p2i[ind],n2i[name]]>0 else "") +
'%s ' % (p2t[ind]) + ' %d' % (implicit_matrix[p2i[ind],:]>0).sum()
 for ind in recommendations ]
# -

[i2t[i] for i in recs[n2i['Rama'],:]]

bm25_matrix.shape

# +
# model.recommend_all?
# -

results = model.similar_items(t2i['Hamburger'],20)
['%s %.4f %d' % (i2t[ind], score, (implicit_matrix[ind,:]>0).sum()) for ind, score in results]

irec = recommenders.ImplicitCollaborativeRecommender(model, bm25_matrix.tocsc())


recs = irec.recommend_all(userids, 20, i2p)

resurface_userids, discovery_userids = wr.load_pickle('../resurface_discovery_users.pickle')


histories_dev = feather.read_feather('../histories_dev.feather')


wr.get_recs_metrics(
    histories_dev, recs, 20, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)

