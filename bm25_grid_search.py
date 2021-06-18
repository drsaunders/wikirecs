from time import time
import wikirecs as wr
import itertools
import implicit
import recommenders
from pyarrow import feather
from implicit.nearest_neighbours import BM25Recommender

histories_test = feather.read_feather("../histories_dev.feather")

implicit_matrix = wr.load_pickle("../implicit_matrix.pickle")
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle(
    "../lookup_tables.pickle"
)

userids, pageids = wr.load_pickle("../users_and_pages.pickle")

resurface_userids, discovery_userids = wr.load_pickle(
    "../resurface_discovery_users.pickle"
)

# regularization_levels = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# num_factors_levels = [10, 50, 100]


K1_levels = [10, 20, 50, 100, 200]
B_levels = [0, 0.25, 0.5, 0.75, 1]
filter_already_liked_items_levels = [True, False]

K = 20

runs = []
for K1, B, filter_already_liked_items in itertools.product(
    K1_levels, B_levels, filter_already_liked_items_levels
):
    print((K1, B, filter_already_liked_items))
    start_time = time()

    model = BM25Recommender(K1, B)
    model.fit(implicit_matrix)

    brec = recommenders.MyBM25Recommender(model, implicit_matrix)
    brecs = brec.recommend_all(
        userids,
        K,
        u2i=u2i,
        n2i=n2i,
        i2p=i2p,
        filter_already_liked_items=filter_already_liked_items,
    )
    print("Computing metrics...")
    metrics = wr.get_recs_metrics(
        histories_test,
        brecs,
        K,
        discovery_userids,
        resurface_userids,
        implicit_matrix,
        i2p,
        u2i,
    )

    run_record = {
        "K1": K1,
        "B": B,
        "metrics": metrics,
        "time": time() - start_time,
    }
    print(run_record)

    runs.append(run_record)

    wr.save_pickle(runs, "../bm25_grid_search.pickle")
