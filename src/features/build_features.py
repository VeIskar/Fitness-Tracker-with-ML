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


#butterworth low-pass filter

df_lowpas = df.copy()
lowpass = LowPassFilter()

fs = 1000/200 #five instances per second
cutoff = 1.3 #(0.4)low number smooth data high(4) -> close to raw data 

df_lowpas = lowpass.low_pass_filter(df_lowpas, "acc_y", fs, cutoff, order=5)

subset = df_lowpas[df_lowpas["set"]==45]
print(subset["label"][0])


#data visualization
fig, ax =plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop = True), label ="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop = True), label ="butterworth filter")
ax[0].legend(loc = "upper center", bbox_to_anchro=(0.5, 1.15), fancybox = True, shadow=True)
ax[1].legend(loc = "upper center", bbox_to_anchro=(0.5, 1.15), fancybox = True, shadow=True)


for col in predictor_cols:
    df_lowpas = LowPassFilter.low_pass_filter(df_lowpas, col , fs, cutoff, order=5)
    #overwriting original column
    df_lowpas[col] = df_lowpas[col+ "_lowpass"]
    del df_lowpas[col+ "_lowpass"]


