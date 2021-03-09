import dask.array as da
x = da.random.random((10000, 10000), chunks=(1000, 1000))

%%time
y = x + x.T
z = y[::2, 5000:].mean(axis=1)
z

%%time
z.compute()

x = np.random.random((10000, 10000))

from math import sqrt
from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))

%%time 
a = [recommender.recommend(userid=u, N=100, interactions=histories_train) for u in userids]

from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))

# %%time 
# from math import sqrt
# from joblib import Parallel, delayed
# a = Parallel(n_jobs=-1)(delayed(recommender.recommend)(u, 100, None, histories_train) for u in userids)