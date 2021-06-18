from time import time
import wikirecs as wr
import itertools
import implicit
import recommenders
from pyarrow import feather
import os
from implicit.nearest_neighbours import bm25_weight

os.environ["OPENBLAS_NUM_THREADS"] = "1"

print("Loading files...")
histories_test = feather.read_feather("../histories_dev.feather")

implicit_matrix = wr.load_pickle("../implicit_matrix.pickle")
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle(
    "../lookup_tables.pickle"
)

userids, pageids = wr.load_pickle("../users_and_pages.pickle")

resurface_userids, discovery_userids = wr.load_pickle(
    "../resurface_discovery_users.pickle"
)

regularization = 0.01
num_factors = 500

K1_levels = [100]
B_levels = [0, 0.25, 0.5, 0.75, 1]
use_bm25 = [True]

K = 20

levels_to_test = list(itertools.product(K1_levels, B_levels, use_bm25))
levels_to_test.append([0, 0, False])

print(levels_to_test)

runs = []
for K1, B, use_bm25 in levels_to_test:
    print((K1, B, use_bm25))
    start_time = time()

    if use_bm25:
        matrix_to_use = bm25_weight(implicit_matrix, K1=100, B=0.5)
        matrix_to_use = matrix_to_use.tocsc()
    else:
        matrix_to_use = implicit_matrix

    model = implicit.als.AlternatingLeastSquares(
        factors=num_factors, regularization=regularization
    )
    model.fit(implicit_matrix)
    irec = recommenders.ImplicitCollaborativeRecommender(model, matrix_to_use)
    irecs = irec.recommend_all(userids, K, i2p, filter_already_liked_items=True)
    metrics = wr.get_recs_metrics(
        histories_test,
        irecs,
        K,
        discovery_userids,
        resurface_userids,
        matrix_to_use,
        i2p,
        u2i,
    )

    run_record = {
        "num_factors": num_factors,
        "regularization": regularization,
        "K1": K1,
        "B": B,
        "use_bm25": use_bm25,
        "metrics": metrics,
        "time": time() - start_time,
    }
    print(run_record)

    runs.append(run_record)

    wr.save_pickle(runs, "../implicit_grid_search_bm25_disc.pickle")
