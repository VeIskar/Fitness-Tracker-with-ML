import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl



#loading data
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


#checking particiapnt name in case of wrong formatting
if df['participant'].str.contains(r'MetaMotion\\').any():
        df['participant'] = df['participant'].str.extract(r'MetaMotion\\(\w)')
        print("participant labels corrected")
else:
        print("participant labels are correct")

# Checking for anomalies and datetime index errors
assert df.index.is_monotonic_increasing, "Datetime index is not monotonic increasing"

df.index = pd.to_datetime(df.index, errors='coerce')  #coercing errors to NaT
df = df.dropna(subset=['acc_y']) #drops rows with NaT

#single columns plots
set_sub_df = df[df["set"] == 1]
plt.plot(set_sub_df["acc_y"]) #with timestamps
plt.plot(set_sub_df["acc_y"].reset_index(drop=True)) #without timestamps

#exercisess plots

for lab in df["label"].unique():
    subset = df[df["label"] == lab]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(set_sub_df[:100]["acc_y"].reset_index(drop=True), label = lab)
    plt.title(f"acc_y values for: {lab}")
    plt.ylabel("acc_y")
    plt.legend()
    plt.show()



#changing plot settings
plt.style.use('seaborn-v0_8-darkgrid')

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

all_axis_df = df.query(f"label == '{label}' and participant == '{participant}'").reset_index()

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
            df.query(f"label == '{lbl}'")
            .query(f"participant == '{prt}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{lbl} ({prt})".title())
            plt.legend()
            #plt.show()  to show

#for gyroscope data
for lbl in labels:
    for prt in participants:
        all_axis_df = (
            df.query(f"label == '{lbl}'")
            .query(f"participant == '{prt}'")
            .reset_index()
        )


        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{lbl} ({prt})".title())
            plt.legend()
            #plt.show() to show



#combining into one figure

label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

#combined accelerometer data
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)


#combined gyroscope data
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].set_xlabel("samples")

#plt.show()


#exporting figures of all combinations of both sensors

labels = df["label"].unique()
participants = df["participant"].unique()

for lbl in labels:
    for prt in participants:
        combined_plot_df = (
            df.query(f"label == '{lbl}'")
            .query(f"participant == '{prt}'")
            .reset_index(drop=True)
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

            #accelerometer data
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            ax[0].set_ylabel("Accelerometer")
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

            #gyroscope data
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
            ax[1].set_ylabel("Gyroscope")
            ax[1].set_xlabel("Samples")
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            
            #save the plot
            plt.savefig(f"../../reports/figures/{label.title()}_({participant}).png")
            #plt.show() show all the plots


