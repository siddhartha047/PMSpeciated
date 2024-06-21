import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
#scale = StandardScaler()
####################################### USEPA DATA IMPORT ##################################################################
#******************************************************************************************************************************#

USEPA_SPECIATE_PATH = 'speciate08132023.csv'
IMPROVE_SPECIES_LIST_PATH = 'improve_params.csv'
USEPA_SPECIATE_ASSIGNED_PROFILE = 'profile_assigned_forSEPCIATEdata.csv'
######## open csv for USEPA SPECIATE data
file = open(USEPA_SPECIATE_PATH,newline='')
reader = csv.reader(file)

header1 = next(reader) # the first line is the header
data = [row for row in reader] # read teh remaining data
file.close()

#print(header1)
#print(data[0])

df = pd.DataFrame(data,columns =header1,dtype='float64')


######## open IMPROVE SPECIES list file

file = open(IMPROVE_SPECIES_LIST_PATH,newline='')
reader = csv.reader(file)

header2 = next(reader) # the first line is the header
#data2 = []
species_name = []
for row in reader:
    str1 = row[0].strip()
    species_name.append(str1)
    #data2.append([species_name,species_code])
file.close()

#print(header)
#print(species_name)

####### select USEPA data based on IMRPROVE species
df1 = df[df.species_name.isin(species_name)]
unique_species_lst = df1['species_name'].unique()
#df1.to_csv('usepa_with_improve_species.csv', encoding='utf-8', index=False)

#stats_numeric =  df1[["total_weight","weight_percent","uncertainty_percent"]].describe()  #df1["total_weight","weight_percent","uncertainty_percent"]
#print(stats_numeric)

#df2= df1.drop(["total_weight","uncertainty_percent"], axis=1)

#print(df2[:5])
df_pivot = df1.pivot(index=['profile_code'],columns='species_name', values='weight_percent').reset_index()
#print(df_pivot[:5])

df_pivot.to_csv('usepa_with_improve_species_pivot.csv', encoding='utf-8', index=False)


##### read assigned profile csv file
file = open(USEPA_SPECIATE_ASSIGNED_PROFILE,newline='')
reader = csv.reader(file)

header3 = next(reader) # the first line is the header
data = [row for row in reader] # read teh remaining data
file.close()

#print(header3)
#print(data[0])

df2 = pd.DataFrame(data,columns =header3)

##### merge main dataframe with assigned profile
df_final0 = pd.merge(df_pivot, df2, on ='profile_code', how ='left')

df_final1 = df_final0.fillna(0)

df_final1.to_csv('usepa_final_with_assigned_profile.csv', encoding='utf-8', index=False)
#null_rows = df_final0.loc[df_final0['assigned_profile']].isnull()
df_final1['max'] = df_final1.max(axis=1)
dff= df_final1.loc[ ( (df_final1['max'] > 0) & (df_final1['assigned_profile'] != "Other")  & (df_final1['assigned_profile'] != "Gas Combustion") & (df_final1['assigned_profile'] != "Aircraft") & (df_final1['assigned_profile'] != "CNG Vehicle") & (df_final1['assigned_profile'] != "Brake Tire Wear") & (df_final1['assigned_profile'] != "Volcano") & (df_final1['assigned_profile'] != "Vehicle Composite"))] #& (df_final1['assigned_profile'] != "Marine vessel")
#lst = dff["assigned_profile"].values.tolist()
dff.index = range(len(dff))
unique_profile_lst = dff['assigned_profile'].unique()
#print(dff[:3])
#t1 = dff[dff.assigned_profile=='Wildfire']
################################################################################################################################
#******************************************************************************************************************************#
#sns.pairplot(t1[t1.columns[1:]],hue='assigned_profile')
#plt.show()

import torch
import torch.nn as nn

#data = torch.tensor( dff[dff.columns[1:38]].values ).float()

data_np = dff[dff.columns[1:38]].to_numpy()

#data_np_norm = data_np#scale.fit_transform(data_np)
data_np_norm = np.divide(data_np,data_np.max(axis=None))  #- divide by array maximum

## convert numpy to tensor
data = torch.tensor(data_np_norm).float()

# transform species to number
labels = torch.zeros(len(data), dtype=torch.long)
for i, ele in enumerate(unique_profile_lst):
    labels[dff.assigned_profile==ele] = i
#    print(ele)
#####################################################################################################################################
###************************************************************************************************************************
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.20)
# https://www.w3schools.com/python/python_ml_scale.asp
#from sklearn import linear_model
#model = linear_model.LinearRegression()
#from sklearn.svm import SVC
#model = SVC()
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
#model.fit(data, labels)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
scores = metrics.accuracy_score(Y_test,Y_pred)
result = metrics.classification_report(Y_test,Y_pred)   
#########################################################################################
#########################################################################################################
#############################################################################################################



#########################################################

# import PMF data
PMF_PATH = 'PMF_PudgetSound2_7Factors.csv'  #  Already normalized #

file = open(PMF_PATH,newline='')
reader = csv.reader(file)

header1 = next(reader) # the first line is the header
data = [row for row in reader] # read teh remaining data
file.close()

df = pd.DataFrame(data,columns =header1)
df = df.apply(pd.to_numeric)

data_np1 = df.to_numpy()
data1 = torch.tensor(data_np1).float()

#scaled = np.array([8.59E-04,5.93E-06,8.10E-04,1.65E-04,1.17E-03,0.00E+00,3.45E-04,2.10E-01,2.41E-01,1.03E-02,0.00E+00,2.38E-03,2.10E-04,0.00E+00,8.14E-06,0.00E+00,0.00E+00,4.99E-04,1.10E+00,6.50E-02,2.62E-01,4.38E-01,2.32E-01,1.55E-04,1.21E-02,0.00E+00,9.03E-06,3.53E-06,1.14E-03,7.25E-04,1.91E-05,5.46E-02,2.07E-02,1.31E-04,0.00E+00,8.22E-04,3.09E-05])
#scale.transform([[1.61E-03,1.43E-05,1.39E-05,1.38E-03,7.86E-05,0.00E+00,3.36E-04,4.44E-02,8.19E-02,1.59E-02,2.54E-06,3.02E-03,2.50E-04,0.00E+00,1.39E-05,0.00E+00,0.00E+00,1.33E-03,5.78E-02,1.04E-03,0.00E+00,8.70E-03,9.79E-03,2.91E-05,7.56E-03,5.20E-02,2.30E-05,2.72E-05,3.99E-03,0.00E+00,2.78E-05,3.52E-02,1.60E-02,2.72E-04,8.26E-06,9.25E-04,1.78E-04]])
predictions = model.predict(data1)
print(predictions)
#lst1=predictions.tolist()
