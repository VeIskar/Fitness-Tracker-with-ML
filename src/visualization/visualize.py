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
participants_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


#multiple plot axis

label = "squat"

participant = "A" 
#all_axis_df = df.query(f"label == '{participant}'").reset_index()
all_axis_df = df.query(f"label == '{label}'").query(f"label == '{participant}'").reset_index()


#data for squatting exercise of participant A in all the 3 axis
#basically movement pattern of the person doing the exercise is displayed through plots
fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


#loop for plots with combinations per sensor

labels = df["label"].unique()
participants = df["participant"].unique()

for lbl in labels:
    for prt in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"label == '{participant}'")
            .reset_index()

        )

        if len(all_axis_df)>0:

            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
    


#for gyroscope data
for lbl in labels:
    for prt in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"label == '{participant}'")
            .reset_index()

        )

        if len(all_axis_df)>0:

            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()



#combining into one figure

label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"label == '{participant}'")
    .reset_index(drop = True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])


ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol =3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol =3, fancybox=True, shadow=True)
ax[1].set_xlabel("samples")
