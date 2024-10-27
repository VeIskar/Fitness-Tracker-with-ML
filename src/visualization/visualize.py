import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl



#loading data
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


#single columns plots
set_sub_df = df[df["set"] == 1]
plt.plot(set_sub_df["acc_y"]) #without timestamps
plt.plot(set_sub_df["acc_y"].reset_index(drop=True)) #with timestamps

#exercisess plots

for lab in df["label"].unique():
    subset = df[df["label"] == lab]

    fig, ax = plt.subplots()
    plt.plot(set_sub_df[:100]["acc_y"].reset_index(drop=True), label = lab)
    plt.legend()
    plt.show()

