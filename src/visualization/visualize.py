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



#changing plot settings
plt.style.use('seaborn-darkgrid')

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100


#comparing heavy and medium sets
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()   #plot with differences between medium and heavy
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


#participants comparison
participants_df = category_df = df.query("label == 'bench'").sort_values("participant").reset_index()


fig, ax = plt.subplots()
participants_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()