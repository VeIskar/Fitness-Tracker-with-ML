import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


#load data
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")


