import os
import pandas as pd
from sklearn.model_selection import train_test_split


def Load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'creditcard.csv')

    df = pd.read_csv(data_path)

    return df


def CheckClassDistribution(df):
    class_counts = df["Class"].value_counts()
    class_percentage = df["Class"].value_counts(normalize=True) * 100

    print("Class Counts:")
    print(class_counts)

    print("\nClass Percentage:")
    print(class_percentage)

    return class_counts, class_percentage


def CheckFeature(df):
    print("Features:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)

    return df.columns, df.dtypes


def CheckFeatureAmount(df):
    print("Amount Statistics:")
    print(df["Amount"].describe())

    return df["Amount"].describe()


def CheckFeatureTime(df):
    print("Time Statistics:")
    print(df["Time"].describe())

    return df["Time"].describe()


def CheckMissingValue(df):
    missing_value = df.isnull().sum()

    print("Missing Values:")
    print(missing_value)

    return missing_value


def CheckDuplicate(df):
    duplicate_count = df.duplicated().sum()

    print("Number of Duplicates:")
    print(duplicate_count)

    return duplicate_count


def SplitData(df):
    x = df.drop("Class", axis=1)
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return x_train, x_test, y_train, y_test