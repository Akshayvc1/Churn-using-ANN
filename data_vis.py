import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlb
import seaborn as sns

data = pd.read_csv('data.csv')

#Potting a bar graph to show the number of true and false values in the dataset
y = data['churn'].value_counts()
sns.barplot(y.index,y.values)

#Plotting bar graphs to show churn by each feature

#Churn by state
data.groupby(["state", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10)) 

#Churn by area code
data.groupby(["area code", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 

#Churn by customers with international plan
data.groupby(["international plan", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 

#Churn by customers with voice mail plan
data.groupby(["voice mail plan", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 
