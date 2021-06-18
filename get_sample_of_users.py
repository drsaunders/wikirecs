import pandas as pd
import numpy as np
import requests
import argparse
from tqdm import tqdm


def get_recent_changes(N):
    S = requests.Session()

    t = tqdm(total=N)

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "format": "json",
        "rcprop": "title|ids|sizes|flags|user|userid|timestamp",
        "rcshow": "!bot|!anon|!minor",
        "rctype": "edit",
        "rcnamespace": "0",
        "list": "recentchanges",
        "action": "query",
        "rclimit": "500",
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    RECENTCHANGES = DATA["query"]["recentchanges"]
    all_rc = RECENTCHANGES

    i = 500
    t.update(500)
    while i <= N:
        last_continue = DATA["continue"]
        PARAMS.update(last_continue)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        RECENTCHANGES = DATA["query"]["recentchanges"]
        all_rc.extend(RECENTCHANGES)
        i = i + 500
        t.update(500)

    if len(all_rc) > N:
        all_rc = all_rc[:N]

    return all_rc


def get_sample_of_users(edit_lookback, outfile=None):
    """Get a sample of recently active users by pulling the most recent N edits
    Note that this will be biased towards highly active users.

    Args:
        edit_lookback: The number of edits to go back.
        outfile: Pickle file path to write the user list to

    Returns:
        Dataframe with user and user id columns
    """
    df = get_recent_changes(edit_lookback)

    # Drop missing userid entries
    df = pd.DataFrame(df).dropna(subset=["userid"])

    print("Earliest timestamp: {}".format(df.timestamp.min()))
    print("Latest timestamp: {}".format(df.timestamp.max()))
    print("Number of distinct users: {}".format(len(df.user.unique())))
    print(
        "Mean number of edits per user in timeframe: %.2f"
        % (len(df) / len(df.user.unique()))
    )
    print("Number of distinct pages edited: {}".format(len(df.pageid.unique())))
    print(
        "Mean number of edits per page in timeframe: %.2f"
        % (len(df) / len(df.pageid.unique()))
    )

    # Deduplicate to get
    sampled_users = df.loc[:, ["user", "userid"]].drop_duplicates()

    # Remove RFD
    sampled_users = sampled_users[np.invert(sampled_users.user == "RFD")]
    sampled_users = sampled_users.reset_index(drop=True)

    if outfile:
        sampled_users.to_csv(outfile, index=False)

    return sampled_users


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edit_lookback",
        type=int,
        required=True,
        help="How many edits to look back in Wikipedia history",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="CSV file to write the resulting user names and ids to",
    )
    args = parser.parse_args()

    get_sample_of_users(args.edit_lookback, args.outfile)