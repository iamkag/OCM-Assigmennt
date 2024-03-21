import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')


def main():

    data = pd.read_csv("analysis.csv")
    #will display the top 5 observations of the dataset
    data.head()
    #will display the last 5 observations of the dataset
    data.tail()
    #info() helps to understand the data type and information about data, including the number of records in each column, data having null or not null, Data type, the memory usage of the dataset
    data.info()
    #Check for Duplication
    data.nunique()
    #Missing Values Calculation
    data.isnull().sum()
    #calculate the percentage of missing values in each column
    (data.isnull().sum()/(len(data)))*100