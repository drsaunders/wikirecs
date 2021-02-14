import wikirecs as wr
import numpy as np
from tqdm.auto import tqdm


class Recommender(object):
    def __init__(self):
        raise NotImplementedError

    def recommend(self, userid=None, username=None, N=10):
        raise NotImplementedError

    def recommend_all(self, userids, num_recs, **kwargs):
        recs = {}
        with tqdm(total=len(userids)) as progress:
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

    def recommend(self, N=10, userid=None, user=None, interactions=None):
        if user is not None:
            is_user_row = interactions.user == user
        elif userid is not None:
            is_user_row = interactions.userid == userid
        else:
            raise ValueError("Either user or userid must be non-null")

        recs = (
            interactions[is_user_row]
            .drop_duplicates(subset=["pageid"])
            .iloc[:N]
            .pageid.values
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

    def recommend(self, N=10, userid=None, user=None, u2i=None, n2i=None, i2p=None):
        if user is not None:
            user_index = n2i[user]
        elif userid is not None:
            user_index = u2i[userid]
        else:
            raise ValueError("Either user or userid must be non-null")

        recs_indices = self.model.recommend(user_index, self.implicit_matrix, N)
        recs = [i2p[a[0]] for a in recs_indices]

        return recs
