import pandas as pd
import numpy as np
import requests
import time
import argparse
from tqdm import tqdm
from pyarrow import feather


def get_edit_history(
    userid=None, user=None, latest_timestamp=None, earliest_timestamp=None, limit=None
):
    """For a particular user, pull their whole history of edits.

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.
    """

    S = requests.Session()
    S.headers.update(
        {"User-Agent": "WikiRecs (danielrsaunders@gmail.com) One-time pull"}
    )

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "ucnamespace": "0",
        "list": "usercontribs",
        "ucuserids": userid,
        "ucprop": "title|ids|sizediff|flags|comment|timestamp",
        "ucshow=": "!minor|!new",
    }
    if latest_timestamp is not None:
        PARAMS["ucstart"] = latest_timestamp
    if earliest_timestamp is not None:
        PARAMS["ucend"] = earliest_timestamp
    if user is not None:
        PARAMS["ucuser"] = user
    if userid is not None:
        PARAMS["ucuserid"] = userid

    PARAMS["uclimit"] = 500

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    if "query" not in DATA:
        print(DATA)
        raise ValueError

    USERCONTRIBS = DATA["query"]["usercontribs"]
    all_ucs = USERCONTRIBS
    i = 500
    while i < 100000:
        if "continue" not in DATA:
            break
        last_continue = DATA["continue"]
        PARAMS.update(last_continue)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        USERCONTRIBS = DATA["query"]["usercontribs"]
        all_ucs.extend(USERCONTRIBS)
        i = i + 500

    return all_ucs


def pull_edit_histories(
    sampled_users_file,
    edit_histories_file_pattern,
    users_per_chunk,
    earliest_timestamp,
    start=0,
):
    histories = []
    cols = ["userid", "user", "pageid", "title", "timestamp", "sizediff"]
    sampled_users = pd.read_csv(sampled_users_file)
    sampled_users.loc[:, "userid"].astype(int)

    sampled_users = sampled_users.reset_index()

    # Iterate through all the users in the list
    for i, (user, userid) in tqdm(
        iterable=enumerate(
            zip(sampled_users["user"][start:], sampled_users["userid"][start:]),
            start=start,
        ),
        total=len(sampled_users),
        initial=start,
    ):
        # Get the history of edits for this userid
        thehistory = get_edit_history(
            userid=int(userid), earliest_timestamp=earliest_timestamp
        )

        # If no edits, skip
        if len(thehistory) == 0:
            continue

        thehistory = pd.DataFrame(thehistory)

        # Remove edits using automated tools by looking for the word "using" in the comments
        try:
            thehistory = thehistory[
                np.invert(thehistory.comment.astype(str).str.contains("using"))
            ]
        except AttributeError:
            continue

        if len(thehistory) == 0:
            continue

        histories.append(thehistory.loc[:, cols])

        if np.mod(i, 50) == 0:
            print(
                "Most recent: {}/{} {} ({}) has {} edits".format(
                    i, len(sampled_users), user, int(userid), len(thehistory)
                )
            )

        # Every x users save it out, for the sake of ram limitations
        if np.mod(i, users_per_chunk) == 0:
            feather.write_feather(
                pd.concat(histories), edit_histories_file_pattern.format(i)
            )

            histories = []

    # Get the last few users that don't make up a full chunk
    feather.write_feather(pd.concat(histories), edit_histories_file_pattern.format(i))


if __name__ == "__main__":

    # pull_edit_histories(
    #     "repo/sampled_users_2021-05-28.csv",
    #     "repo/edit_histories_2021-05-28_{}.feather",
    #     5,
    #     "2020-05-12T02:59:44Z",
    # )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampled_users_file",
        type=str,
        required=True,
        help="CSV file with the list of users",
    )
    parser.add_argument(
        "--edit_histories_file_pattern",
        type=str,
        required=True,
        help="Output for edit histories (needs a {} for where to place the number)",
    )
    parser.add_argument(
        "--users_per_chunk",
        type=int,
        required=True,
        help="How many users to pull before dumping to a file (for the sake of ram)",
    )
    parser.add_argument(
        "--earliest_timestamp",
        type=str,
        required=True,
        help="How far back to go (format e.g. 2021-02-11T05:35:34Z)",
    )
    args = parser.parse_args()

    pull_edit_histories(
        args.sampled_users_file,
        args.edit_histories_file_pattern,
        args.users_per_chunk,
        args.earliest_timestamp,
        start=0,
    )
