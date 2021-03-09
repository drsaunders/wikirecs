from repo.recommenders import JaccardRecommender
import numpy as np
from scipy.sparse.csc import csc_matrix


def test_jaccard():
    implicit_matrix = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 0]])
    assert implicit_matrix.shape == (3, 4)
    implicit_matrix = csc_matrix(implicit_matrix)
    n2i = {"huey": 0, "dewey": 1, "louie": 2, "chewy": 3}
    t2i = {"Batman": 0, "Mystery Men": 1, "Taxi Driver": 2}
    i2n = {v: k for k, v in n2i.items()}
    i2t = {v: k for k, v in t2i.items()}

    jrec = JaccardRecommender(implicit_matrix, p2i=None, t2i=t2i, i2t=i2t, i2p=None)

    print(jrec.item_to_item(N=10, title="Batman"))


if __name__ == "__main__":
    test_jaccard()