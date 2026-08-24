import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def Load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'creditcard.csv')
    df = pd.read_csv(data_path)
    return df

def dataset_info(df):
    print("dataset shape:\n")
    print(df.shape)
    print("dataset info:")
    print(df.info())

def CheckClassDistribution(df):
    class_counts = df["Class"].value_counts()
    class_percentage = df["Class"].value_counts(normalize=True) * 100

    print("\nClass Percentage:")
    print(class_percentage)

    return  class_percentage


def CheckFeature(df):
    print("Features:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)



def CheckFeatureAmount(df):
    print("Amount Statistics:")
    print(df["Amount"].describe())



def CheckFeatureTime(df):
    print("Time Statistics:")
    print(df["Time"].describe())



def CheckMissingValue(df):
    missing_value = df.isnull().sum()

    print("Missing Values:")
    print(missing_value)



def CheckDuplicate(df):
    duplicate_count = df.duplicated().sum()

    print("Number of Duplicates:")
    print(duplicate_count)
    print("remove duplicates:\n",df.drop_duplicates())

    return df.drop_duplicates()
