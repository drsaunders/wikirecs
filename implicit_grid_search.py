from time import time
import wikirecs as wr
import itertools
import implicit
import recommenders
from pyarrow import feather

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


regularization_levels = [0.001, 0.01, 0.1]
num_factors_levels = [250, 500]


K = 20

runs = []
for num_factors, regularization in itertools.product(
    num_factors_levels, regularization_levels
):
    print((num_factors, regularization))
    start_time = time()

    model = implicit.als.AlternatingLeastSquares(
        factors=num_factors, regularization=regularization
    )
    model.fit(implicit_matrix)
    irec = recommenders.ImplicitCollaborativeRecommender(model, implicit_matrix)
    irecs = irec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p)
    metrics = wr.get_recs_metrics(
        histories_test,
        irecs,
        K,
        discovery_userids,
        resurface_userids,
        implicit_matrix,
        i2p,
        u2i,
    )

    run_record = {
        "num_factors": num_factors,
        "regularization": regularization,
        "metrics": metrics,
        "time": time() - start_time,
    }
    print(run_record)

    runs.append(run_record)

    wr.save_pickle(runs, "../implicit_grid_search.pickle")
