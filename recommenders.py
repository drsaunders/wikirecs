import wikirecs as wr
import numpy as np
from tqdm.auto import tqdm
import itertools
import pandas as pd
from implicit.nearest_neighbours import BM25Recommender


class Recommender(object):
    def __init__(self):
        raise NotImplementedError

    def recommend(self, userid=None, username=None, N=10):
        raise NotImplementedError

    def recommend_all(self, userids, num_recs, **kwargs):
        recs = {}
        with tqdm(total=len(userids), leave=True) as progress:
            for u in userids:
                recs[u] = self.recommend(userid=u, N=num_recs, **kwargs)
                progress.update(1)

        return recs


class PopularityRecommender(Recommender):
    def __init__(self, interactions):
        with wr.Timer("Building popularity table"):
            self.editors_per_page = (
                interactions.drop_duplicates(subset=["title", "user"])
                .groupby(["pageid", "title"])
                .count()
                .user.sort_values(ascending=False)
            )

    def recommend(self, N=10, userid=None, user=None):
        return self.editors_per_page.iloc[:N].index.get_level_values(0).values


class MostRecentRecommender(Recommender):
    """
    Recommend the most recently edited pages by the user in reverse chronological
    order. When those run out, go to most popular
    """

    def __init__(self, interactions):
        with wr.Timer("Building popularity table"):
            self.editors_per_page = (
                interactions.drop_duplicates(subset=["title", "user"])
                .groupby(["pageid", "title"])
                .count()
                .user.sort_values(ascending=False)
            )

    def all_recent_only(self, N=10, userids=None, interactions=None):
        recents = {}

        with tqdm(total=len(userids), leave=True) as progress:
            for u in userids:
                is_user_row = interactions.userid == u
                recents[u] = (
                    interactions[is_user_row]
                    .drop_duplicates(subset=["pageid"])
                    .iloc[:N]
                    .pageid.values
                )
                progress.update(1)
        return recents

    def recommend(self, N=10, userid=None, user=None, interactions=None):
        if user is not None:
            is_user_row = interactions.user == user
        elif userid is not None:
            is_user_row = interactions.userid == userid
        else:
            raise ValueError("Either user or userid must be non-null")

        deduped_pages = interactions[is_user_row].drop_duplicates(subset=["pageid"])
        if len(deduped_pages) == 1:
            recs = []
        else:
            # Don't take the most recent, because this dataset strips out repeated instance
            recs = deduped_pages.iloc[1:N].pageid.values

        # If we've run out of recs, fill the rest with the most popular entries
        if len(recs) < N:
            recs = np.concatenate(
                [
                    recs,
                    self.editors_per_page.iloc[: (N - len(recs))]
                    .index.get_level_values(0)
                    .values,
                ]
            )
        return recs


class MostFrequentRecommender(Recommender):
    """
    Recommend the most frequently edited pages by the user. When those run out, go to most popular
    """

    def __init__(self, interactions):
        with wr.Timer("Building popularity table"):
            self.editors_per_page = (
                interactions.drop_duplicates(subset=["title", "user"])
                .groupby(["pageid", "title"])
                .count()
                .user.sort_values(ascending=False)
            )

    def recommend(self, N=10, userid=None, user=None, interactions=None):
        if user is not None:
            is_user_row = interactions.user == user
        elif userid is not None:
            is_user_row = interactions.userid == userid
        else:
            raise ValueError("Either user or userid must be non-null")

        recs = (
            interactions[is_user_row]
            .groupby("pageid")
            .user.count()
            .sort_values(ascending=False)
            .index[:N]
            .values
        )

        # If we've run out of recs, fill the rest with the most popular entries
        if len(recs) < N:
            recs = np.concatenate(
                [
                    recs,
                    self.editors_per_page.iloc[: (N - len(recs))]
                    .index.get_level_values(0)
                    .values,
                ]
            )
        return recs


class ImplicitCollaborativeRecommender(Recommender):
    def __init__(self, model, implicit_matrix):
        self.model = model
        self.implicit_matrix = implicit_matrix

    def recommend(
        self,
        N=10,
        userid=None,
        user=None,
        u2i=None,
        n2i=None,
        i2p=None,
        filter_already_liked_items=False,
    ):
        if user is not None:
            user_index = n2i[user]
        elif userid is not None:
            user_index = u2i[userid]
        else:
            raise ValueError("Either user or userid must be non-null")

        recs_indices = self.model.recommend(
            user_index,
            self.implicit_matrix,
            N,
            filter_already_liked_items=filter_already_liked_items,
        )
        recs = [i2p[a[0]] for a in recs_indices]

        return recs

    def recommend_all(self, userids, num_recs, i2p, filter_already_liked_items=True):
        all_recs = self.model.recommend_all(
            self.implicit_matrix.T,
            num_recs,
            filter_already_liked_items=filter_already_liked_items,
        )
        recs = {
            userid: [i2p[i] for i in all_recs[i, :]] for i, userid in enumerate(userids)
        }

        return recs


class MyBM25Recommender(Recommender):
    def __init__(self, model, implicit_matrix):
        self.model = model

        self.implicit_matrix = implicit_matrix

    def recommend(
        self,
        N=10,
        filter_already_liked_items=True,
        userid=None,
        user=None,
        u2i=None,
        n2i=None,
        i2p=None,
    ):
        if user is not None:
            user_index = n2i[user]
        elif userid is not None:
            user_index = u2i[userid]
        else:
            raise ValueError("Either user or userid must be non-null")

        recs_indices = self.model.recommend(
            user_index,
            self.implicit_matrix.astype(np.float32),
            N,
            filter_already_liked_items=filter_already_liked_items,
        )
        recs = [i2p[a[0]] for a in recs_indices]

        if len(recs) <= 20:
            recs = recs + [recs[-1]] * (20 - len(recs))

        return recs


class JaccardRecommender(Recommender):
    def __init__(self, implicit_matrix, p2i, t2i, i2t, i2p, n2i, u2i, i2u):
        self.implicit_matrix = implicit_matrix
        self.p2i = p2i
        self.t2i = t2i
        self.i2t = i2t
        self.i2p = i2p
        self.n2i = n2i
        self.i2p = i2p
        self.u2i = u2i
        self.i2u = i2u

    def jaccard_multiple(self, page_indices, exclude_index=None):
        X = self.implicit_matrix.astype(bool).astype(int)
        if exclude_index is None:
            intrsct = X.dot(X[page_indices, :].T)
            totals = X[page_indices, :].sum(axis=1).T + X.sum(axis=1)
        else:
            use_indices = np.full(X.shape[1], True)
            use_indices[exclude_index] = False
            # print(X[:, use_indices].shape)
            # print(X[page_indices, :][:, use_indices].T.shape)

            intrsct = X[:, use_indices].dot(X[page_indices, :][:, use_indices].T)
            totals = X[page_indices, :][:, use_indices].sum(axis=1).T + X[
                :, use_indices
            ].sum(axis=1)

        return intrsct / (totals - intrsct)

    def recommend(
        self,
        N=10,
        userid=None,
        user=None,
        num_lookpage_pages=None,
        recent_pages_dict=None,
        interactions=None,
    ):
        if user is not None:
            user_index = self.n2i[user]
        elif userid is not None:
            user_index = self.u2i[userid]
        else:
            raise ValueError("Either user or userid must be non-null")

        recent_pages = recent_pages_dict[self.i2u[user_index]][:num_lookpage_pages]

        user_page_indices = [self.p2i[p] for p in recent_pages]
        d = self.jaccard_multiple(user_page_indices, exclude_index=user_index)

        d = np.nan_to_num(d)
        d[d == 1] = np.nan

        mean_jaccard = np.nanmean(d, axis=1).A.squeeze()
        order = np.argsort(mean_jaccard)[::-1]
        return [self.i2p[o] for o in order[:N]]

    def item_to_item(self, N=10, title=None, pageid=None):
        if title is not None:
            page_index = self.t2i.get(title, None)
        elif pageid is not None:
            page_index = self.p2i.get(pageid, None)
        else:
            raise ValueError("Either title or pageid must be non-null")

        if page_index is None:
            raise ValueError(
                "Page {} not found".format(pageid if title is None else title)
            )

        target_page_editors = np.flatnonzero(
            self.implicit_matrix[page_index, :].toarray()
        )
        # print("target_page_editors {}".format(target_page_editors))

        num_target_editors = len(target_page_editors)

        edited_indices = np.flatnonzero(
            np.sum(self.implicit_matrix[:, target_page_editors] > 0, axis=1)
        )

        # print("edited_indices {}".format(edited_indices))

        num_shared_editors = np.asarray(
            np.sum(self.implicit_matrix[:, target_page_editors] > 0, axis=1)[
                edited_indices
            ]
        ).squeeze()

        # print("num_shared_editors {}".format(num_shared_editors))

        num_item_editors = np.asarray(
            np.sum(self.implicit_matrix[edited_indices, :] > 0, axis=1)
        ).squeeze()

        # print("num_item_editors {}".format(num_item_editors))
        # print("Type num_item_editors {}".format(type(num_item_editors)))
        # print("num_item_editors dims {}".format(num_item_editors.shape))

        jaccard_scores = (
            num_shared_editors.astype(float)
            / ((num_target_editors + num_item_editors) - num_shared_editors)
        ).squeeze()

        # print("jaccard_scores {}".format(jaccard_scores))

        sorted_order = np.argsort(jaccard_scores)
        sorted_order = sorted_order.squeeze()

        rec_indices = edited_indices.squeeze()[sorted_order][::-1]
        sorted_scores = jaccard_scores.squeeze()[sorted_order][::-1]
        sorted_num_shared_editors = num_shared_editors.squeeze()[sorted_order][::-1]
        sorted_num_item_editors = num_item_editors.squeeze()[sorted_order][::-1]

        if title is None:
            return list(
                zip(
                    [self.i2p[i] for i in rec_indices[:N]],
                    sorted_scores[:N],
                    sorted_num_shared_editors[:N],
                    sorted_num_item_editors[:N],
                )
            )
        else:
            return list(
                zip(
                    [self.i2t[i] for i in rec_indices[:N]],
                    sorted_scores[:N],
                    sorted_num_shared_editors[:N],
                    sorted_num_item_editors[:N],
                )
            )


class InterleaveRecommender(Recommender):
    """
    Recommend for users by interleaving recs from multiple lists. When there is
    duplicates keeping only the first instance.
    """

    def __init__(self):
        pass

    def recommend_all(self, N=10, recs_list=[]):
        """
        Args:
            N (int): Number of recs to return
            recs_list: Array of recs, which are ordered lists of pageids in a dict keyed by a userid

        Returns:
            dict: Recommendations, as a list of pageids keyed by userid
        """

        def merge_page_lists(page_lists):
            return pd.unique(list(itertools.chain(*zip(*page_lists))))

        return {
            userid: merge_page_lists([recs.get(userid, []) for recs in recs_list])[:N]
            for userid in recs_list[0]
        }
