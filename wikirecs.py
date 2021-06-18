import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
import collections
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score
from tqdm.auto import tqdm


class Timer:
    def __init__(self, segment_label=""):
        self.segment_label = segment_label

    def __enter__(self):
        print(" Entering code segment {}".format(self.segment_label))
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(" Code segment {} took {}".format(self.segment_label, self.interval))


def load_pickle(filename):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def save_pickle(theobject, filename):
    with open(filename, "wb") as fh:
        pickle.dump(theobject, fh)


def conv_wikipedia_timestamp(timestamp):
    return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")


# Look at one user's edits
def print_user_history(all_histories, user=None, userid=None):

    if user is not None:
        edits = all_histories[all_histories.user == user].copy()
    elif userid is not None:
        edits = all_histories[all_histories.userid == userid].copy()
    else:
        raise ValueError("Either user or userid must be non-null")

    if len(edits) == 0:
        print("User not found")

    last_page = -1
    last_date = ""
    for i in range(len(edits))[::-1]:
        row = edits.iloc[i]
        dt = conv_wikipedia_timestamp(row.first_timestamp)
        # Every day of edits, print out the date and reset the "last edited"
        if str(dt.date()) != last_date:
            print(dt.date())
            last_date = str(dt.date())
            last_page = -1

        # Only output when they edit a new page (print only the timestamp of a first of a string of edits)
        if row.pageid != last_page:
            print(" {} {}".format(str(dt.time()), row.title.strip()))
            last_page = row.pageid


def dataframe_set_subtract(df, to_subtract, by_cols=None):
    original_cols = df.columns
    if by_cols is None:
        merge_df = df.merge(to_subtract, how="left", indicator=True)
    else:
        merge_df = df.merge(to_subtract, how="left", indicator=True, on=by_cols)

    return df.loc[merge_df._merge == "left_only", original_cols]


def recall_curve(test_set, recs, max_k, userid_subset=None):
    recall_vals = []
    for K in np.arange(max_k) + 1:
        recall_vals.append(recall(test_set, recs, K, userid_subset))

    return recall_vals


def recall(test_set, recs, K=10, userid_subset=[]):
    """For a test set, compute the % of users who have a hit in the top K.

    Args:
        test_set: DF with an entry for each user with the target edit-to-be-predicted
        recs: Dict by userid of lists of pageid recs
        K: Number of recs to consider when looking for a hit
        userid_subset: Only compute for the userids in this list

    Returns:
        float of the mean number of test entries with hits in the top K
    """

    if userid_subset is None:
        userid_subset = []
    userid_subset = set(userid_subset)

    hits = [
        pageid in recs[userid][:K]
        for pageid, userid in zip(test_set.pageid, test_set.userid)
        if (len(userid_subset) == 0) or (userid in userid_subset)
    ]

    return np.mean(hits)


def prep_for_metrics(test_set, recs, K, userid_subset=None):
    test_set = test_set.drop(columns=["recs"], errors="ignore")
    test_set = test_set.merge(
        pd.DataFrame(
            [(u, recs[u]) for u in test_set.userid], columns=["userid", "recs"]
        ),
        on="userid",
    )
    if userid_subset is None:
        selected_rows = [True] * len(test_set)
    else:
        selected_rows = test_set.userid.isin(userid_subset)

    y_true = [
        (p == r[:K]).astype(int)
        for p, r in zip(
            test_set[selected_rows].pageid.values, test_set[selected_rows].recs.values
        )
    ]
    dummy_y_score = len(test_set[selected_rows]) * [list(range(K))[::-1]]

    test_set = test_set.drop(columns=["recs"])

    return y_true, dummy_y_score


def ndcg(test_set, recs, K=20, userid_subset=None):
    y_true, dummy_y_score = prep_for_metrics(test_set, recs, K, userid_subset)

    ## Print the individual scores
    # for yt in y_true:
    #     print(
    #         (
    #             yt,
    #             ndcg_score(
    #                 np.array(yt, ndmin=2), np.array(list(range(K))[::-1], ndmin=2)
    #             ),
    #         )
    #     )
    return ndcg_score(y_true, dummy_y_score)


def mrr(test_set, recs, K=20, userid_subset=None):
    y_true, dummy_y_score = prep_for_metrics(test_set, recs, K, userid_subset)
    return label_ranking_average_precision_score(y_true, dummy_y_score)


def get_recs_metrics(
    test_set, recs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i
):
    return {
        "recall": 100 * recall(test_set, recs, K),
        "ndcg": ndcg(test_set, recs, K),
        "resurfaced": 100 * prop_resurface(recs, K, implicit_matrix, i2p, u2i),
        "recall_discover": 100
        * recall(test_set, recs, K, userid_subset=discovery_userids),
        "recall_resurface": 100
        * recall(test_set, recs, K, userid_subset=resurface_userids),
        "ndcg_discover": ndcg(test_set, recs, K, userid_subset=discovery_userids),
        "ndcg_resurface": ndcg(test_set, recs, K, userid_subset=resurface_userids),
    }


def prop_resurface(recs, K=10, implicit_matrix=None, i2p=None, u2i=None):
    """What proportion of the top K recs are resurfaced pages (already edited by user)?

    Args:

    Returns:
        float of the mean number of resurfaced pages in the top K recs
    """
    prop_resurface = []
    for userid in recs.keys():
        past_pages = [i2p[i] for i in implicit_matrix[:, u2i[userid]].nonzero()[0]]
        rec_pages = recs[userid][:K]
        prop_resurface.append(np.mean(np.isin(rec_pages, past_pages)))

    return np.mean(prop_resurface)


def display_recs_with_history(
    recs,
    userids,
    histories_test,
    histories_train,
    p2t,
    u2n,
    recs_to_display=5,
    hist_to_display=10,
):
    """Return a dataframe to display showing the true next edit for each user along
    with the top n recs and the history

    Args:
        recs: List of recs per user
        userids: List of users to show this for
        histories_test: The true next edit for all users
        histories_train: The history of all users minus the test set
        recs_to_display: How many of the top recs to show
        hist_to_display: How many edits to look back
        p2t: Page id to title
        u2n: User id to username

    Returns:
        dataframe for display formatting all this information
    """
    display_dicts = collections.OrderedDict()

    index_labels = (
        ["True value"]
        + ["Rec " + str(a + 1) for a in range(recs_to_display)]
        + ["-"]
        + ["Hist " + str(a + 1) for a in range(hist_to_display)]
    )
    for u in userids:
        real_next = histories_test.loc[histories_test.userid == u].pageid.values[0]
        user_history = list(
            histories_train.loc[histories_train.userid == u].title.values[
                :hist_to_display
            ]
        )

        # If we don't have enough history, pad it out with "-"
        if len(user_history) < hist_to_display:
            user_history = user_history + ["-"] * (hist_to_display - len(user_history))
        display_dicts[u2n[u]] = (
            [p2t[real_next]]
            + [p2t[a] for a in recs[u][:recs_to_display]]
            + ["----------"]
            + user_history
        )

    return pd.DataFrame(display_dicts, index=index_labels)


def get_resurface_discovery(histories_train, histories_test):
    d = histories_train.merge(
        histories_test[["userid", "pageid"]], on="userid", suffixes=["", "target"]
    )
    d.loc[:, "is_target"] = d.pageid == d.pageidtarget
    d = d.groupby("userid").is_target.any()
    resurface_userids = d[d].index.values
    discovery_userids = d[~d].index.values

    return (resurface_userids, discovery_userids)


# def compare_user_recs(userid, recs_list, recs_names):
