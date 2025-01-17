import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn


#data loading
df = pd.read_pickle("../data/interim/01_data_processed.pkl")

#checking particiapnt name in case of wrong formatting
if df['participant'].str.contains(r'MetaMotion\\').any():
        df['participant'] = df['participant'].str.extract(r'MetaMotion\\(\w)')
        print("participant labels corrected")
else:
        print("participant labels are correct")

#outliers plotting

outlier_cols = list(df.columns[:6])

plt.style.use("fivethirtyegiht")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100

df[["acc_x","label"]].boxplot(by="label", figsize =(20,10))
df[["gyr_y","label"]].boxplot(by="label", figsize =(20,10))


#acceloremeter:
df[outlier_cols[:3]+["label"]].boxplot(by="label", 
                                       figsize =(20,10), layout =(1,3))

#now for gyroscope data:
df[outlier_cols[3:]+["label"]].boxplot(by="label", 
                                       figsize =(20,10), layout =(1,3))
#plt.show()



def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

###DISTRIBUTION BASED METHODS:

#marking outliers with IQR (Interquartile range)
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

#testing for a single column
col = "acc_x"
dataset = mark_outliers_iqr(df, col)
#dataset.head()

plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)

#loop over all columns
for col in outlier_cols:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)


#Chauvenet criterion
#According to Chauvenet’s criterion we reject a measurement (outlier) 
#from a dataset of size N when it’s probability of observation is less than 1/2N
#a generalization is to replace the value 2 with a parameter C.

#checking for normal distribution
#look at either histogram or boxplot:
#Histogram — Do you see a bell shaped curve / Boxplot — Is the box symmetrical

#histograms:
df[outlier_cols[:3]+["label"]].plot.hist(by="label", 
                                       figsize =(20,10), layout =(3,3))

df[outlier_cols[3:]+["label"]].plot.hist(by="label", 
                                       figsize =(20,10), layout =(3,3))



def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


#looping over all columns with chauvenet
for col in outlier_cols:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)




#DISTANCE BASED METHOD:

#local outlier factor
#It measures the local deviation of 
#the density of a given sample with respect to its neighbors

#we will look for 20 closest neighbours then check wether values are isolated
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1 #-1 is outlier 1 is not
    return dataset, outliers, X_scores


dataset,outliers, X_scores = mark_outliers_lof(df, outlier_cols)
for col in outlier_cols:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof",reset_index=True)


#checking outliers by label grouping
label = "bench"

for col in outlier_cols:
    dataset = mark_outliers_iqr(df[df["label"]==label], col)
    plot_binary_outliers(dataset, col, col + "_outlier",reset_index=True)


for col in outlier_cols:
    dataset = mark_outliers_chauvenet(df[df["label"]==label], col)
    plot_binary_outliers(dataset, col, col + "_outlier",reset_index=True)


dataset,outliers, X_scores = mark_outliers_lof(df, outlier_cols)
for col in outlier_cols:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof",reset_index=True)


#additonal testing of methods

#single gyroscope column test: 
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col=col)
dataset[dataset["gyr_z_outlier"]]

dataset.loc[dataset["gyr_z_outlier"], "gyr_z"] = np.nan


#loop for all
outlier_removed_df = df.copy()

for col in outlier_cols:
     for label in df["label"].unique():
          #we create a subset of original df based off the for loop label
          dataset = dataset = mark_outliers_chauvenet(df[df["label"]==label], col)
          
          #replace values marked as outliers with NaN
          dataset.loc[dataset[col + "_outlier"], col] = np.nan 

          #update original dataframe column
          outlier_removed_df.loc[(outlier_removed_df["label"]==label), col] = dataset[col]
          #we take output subset then speciyf NaN column

          removed_vals = len(dataset) - len(dataset[col].dropna())
          print(f"removed {removed_vals} from {col} for {label}")
          

#show the replaced values marked as outlier in dataframe
outlier_removed_df.info()


#exporting the dataframe
outlier_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
