"""
Helper script to make a train- and test dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(filename: str) -> None:
    df = pd.read_csv(filename, header=None)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    name_without_extension = filename.replace(".csv", "")
    train_df.to_csv(f"{name_without_extension}_train.csv", index=False, header=None)
    test_df.to_csv(f"{name_without_extension}_test.csv", index=False, header=None)


if __name__ == "__main__":
    make_split("data/iris.csv")
