import pandas as pd
from glob import glob


files = glob("../../data/raw/MetaMotion/*.csv")

#size
print(f"Number of files: {len(files)}")


#reading all files
data_path = ("../../data/raw/MetaMotion/")

def read_data_from_files(files):
    acc_df = pd.DataFrame() #Accelerometer
    gyr_df = pd.DataFrame() #Gyroscope

    acc_set = 1
    gyr_set = 1

    for f in files:
        
        #extracting features from filename
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        
        if "Accelerometer" in f:    #to make Accelerometer dataframe
            df['set'] = acc_set #for further visualizing of sets and easier grouping
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        
        if "Gyroscope" in f:    #to make Gyroscope dataframe
            df['set'] = gyr_set
            gyr_set+=1
            gyr_df = pd.concat([gyr_df, df])
        

    #datetime data

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]


    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


#merging/concatanation of datasets

merged_data = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1) #we choose only first 3 columns from accelerometer

merged_data.dropna()


#renaming column names
merged_data.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]


#data resampling 
sampling = {
    "acc_x":"mean",
    "acc_y":"mean",
    "acc_z":"mean",
    "gyr_x":"mean",
    "gyr_y":"mean",
    "gyr_z":"mean",
    "participant":"last",
    "label":"last",
    "category":"last",
    "set":"last",
}


days_data = [i for j, i in merged_data.groupby(pd.Grouper(freq= "D"))]


#resampled data
data_res = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days_data])
data_res["set"] = data_res["set"].astype("int")


#exporting dataset
#data_res.to_pickle("../../data/interim/01_data_processed.pkl")