import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


#load data
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")


predictor_cols = list(df.columns[:6])


plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


#missing values
#we will interpolate missing values (since they have small value/s) to get rid of them
for col in predictor_cols:
    df[col] = df[col].interpolate()

df.info()

#calculating set duration
df[df["set"]==50]["acc_y"].plot()
duration = df[df["set"]==1].index[-1] - df[df["set"]==1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"]==s].index[0]
    stop = df[df["set"]==s].index[-1]

    dur = stop - start

    df.loc[(df["set"]==s), "duration"] = dur.seconds #creating duration column


dur_df = df.groupby(["category"])["duration"].mean()

#duration for heavy sets:
dur_df.iloc[0]/5

#duration for medium sets
dur_df.iloc[1]/10


