"""
Helper script to make a train- and test dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets


def get_iris_data():
    iris = datasets.load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target], axis=1)

    # Split dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    train_df.to_csv("data/iris_train.csv", index=False, header=None)
    test_df.to_csv("data/iris_test.csv", index=False, header=None)


def get_digit_data():
    digits = datasets.load_digits(as_frame=True)
    df = pd.concat([digits.data, digits.target], axis=1)

    # Split dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    train_df.to_csv("data/digits_train.csv", index=False, header=None)
    test_df.to_csv("data/digits_test.csv", index=False, header=None)


if __name__ == "__main__":
    get_iris_data()
    get_digit_data()
