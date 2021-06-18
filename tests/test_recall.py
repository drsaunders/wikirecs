import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import wikirecs as wr
import pandas as pd


def test_recall_typical():
    typical_test_set = pd.DataFrame(
        {
            "userid": [2, 31, 32, 70, 76],
            "pageid": [18402344, 58812518, 1187097, 129540, 53864279],
        }
    )

    recs = {
        2: [63136688, 10849236, 19167679, 18402344, 19594028, 63055098],
        31: [910445, 33372313, 64833595, 59118602, 56300556, 66093761],
        32: [59799, 5595351, 140899, 27003186, 4494959, 6136828],
        70: [25717, 65930, 2698660, 11376, 249268, 32188],
        76: [67319458, 65564103, 64154311, 64373673, 67321086, 53864279],
    }

    print(wr.recall(typical_test_set, recs, K=5, userid_subset=None))


def test_recall_curve():
    typical_test_set = pd.DataFrame(
        {
            "userid": [2, 31, 32, 70, 76],
            "pageid": [18402344, 58812518, 1187097, 129540, 53864279],
        }
    )

    recs = {
        2: [63136688, 10849236, 19167679, 18402344, 19594028, 63055098],
        31: [910445, 33372313, 64833595, 59118602, 56300556, 66093761],
        32: [59799, 5595351, 140899, 27003186, 4494959, 6136828],
        70: [25717, 65930, 2698660, 11376, 249268, 32188],
        76: [67319458, 65564103, 64154311, 64373673, 67321086, 53864279],
    }

    curve = wr.recall_curve(typical_test_set, recs, 5)
    print(curve)


if __name__ == "__main__":
    test_recall_curve()