#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# In[2]:


dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset.describe()


# 
# # Data PreProcessing
# 

# ### Data Visualization

# In[3]:


dataset.head()


# In[4]:


dataset.columns


# In[5]:


features =['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
X = dataset[features]
Y = dataset['stroke']


# In[6]:


X


# In[7]:


sns.countplot(x="stroke",data=dataset)


# In[8]:


sns.countplot(x="gender",data=dataset)


# In[9]:


sns.countplot(x="hypertension",data=dataset)


# In[10]:


sns.countplot(x="heart_disease",data=dataset)


# In[11]:


sns.countplot(x="ever_married",data=dataset)


# In[12]:


sns.countplot(x="work_type",data=dataset)


# In[13]:


sns.countplot(x="Residence_type",data=dataset)


# In[14]:


sns.countplot(x="smoking_status",data=dataset)


# In[15]:


ax = sns.boxplot(x="stroke", y="bmi", data=dataset)


# In[16]:


ax = sns.boxplot(x="stroke", y="avg_glucose_level", data=dataset)


# In[17]:


ax = sns.boxplot(x="stroke", y="age", data=dataset)


# ### Treating NaN, null Values

# In[18]:


dataset.isnull().sum()


# In[19]:


# Replacing Null Numeric vals

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X.iloc[:,8:9])
X.iloc[:,8:9] = imputer.transform(X.iloc[:,8:9])
print(X.isnull().sum())


# In[20]:


# dropping the rows having NaN values 
# df = pd.DataFrame(X)   
# df = df.dropna()   
# print(df)
plt.figure(figsize = (12, 8))
sns.heatmap(dataset.corr(), linecolor = 'white', linewidths = 1, annot = True)
plt.show()


# In[21]:


X['gender'].value_counts()


# In[22]:


# from sklearn.utils import resample

# train_data = pd.concat([X, Y], axis=1)
# # separate minority and majority classes
# negative = train_data[train_data.stroke==0]
# positive = train_data[train_data.stroke==1]
# # upsample minority
# pos_upsampled = resample(positive,
#  replace=True, # sample with replacement
#  n_samples=len(negative), # match number in majority class
#  random_state=27) # reproducible results
# # combine majority and upsampled minority
# upsampled = pd.concat([negative, pos_upsampled])
# # check new class counts
# upsampled.stroke.value_counts()


# In[23]:


# X = upsampled.iloc[:, 0:10]
# Y = upsampled.iloc[:,10:11]
# del X_train['stroke']
print(X,Y)


# ### Treating Categorical Variables
# 

# In[24]:


X.dtypes
Categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
for col in Categorical_cols:
    print('Column Name: ' +col)
    print(X[col].value_counts())


# In[25]:


# Treating Independent Varibles


# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
# df = np.array(ct.fit_transform(df))

                    #OR

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = pd.DataFrame(OH_encoder.fit_transform(X[Categorical_cols]))

# One-hot encoding removed index; put it back
X_encoded.index = X.index

# Remove categorical columns (will replace with one-hot encoding)
num_X = X.drop(Categorical_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X = pd.concat([num_X, X_encoded], axis=1)

pickle.dump(OH_encoder,open("OH_Encoder.pkl","wb"))
# In[26]:


print(OH_X)


# In[27]:


OH_X.describe()


# In[28]:


Y


# In[29]:


from imblearn.over_sampling import ADASYN
from collections import Counter
counter = Counter(Y)
print('before :',counter)
ADA = ADASYN(random_state=130,sampling_strategy='minority')
OH_X,Y = ADA.fit_resample(OH_X,Y)
counter = Counter(Y)
print("after :",counter)


# ### Splitting Dataset
# 

# In[30]:


from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(OH_X, Y, test_size = 0.2, random_state = 1)


# ### Feature Scaling
# 

# In[31]:


# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_
# print(explained_variance)


# In[32]:


X_train.iloc[:,0:5]


# In[33]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:,0:5] = sc.fit_transform(X_train.iloc[:,0:5])
X_test.iloc[:,0:5] = sc.transform(X_test.iloc[:,0:5])
print(X_train,X_test)
pickle.dump(sc,open('featurescale.pkl','wb'))


# In[ ]:





# # Models
# 

# ## Classifiers

# In[36]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error,confusion_matrix
nodes = [10,25,50,100,300]
accuracy =[]
for node in nodes:    
    model = DecisionTreeClassifier(random_state=1,max_leaf_nodes=node)
    model.fit(X_train,Y_train)
    preds = model.predict(X_test)
    accs = accuracy_score(Y_test,preds.round())
    accuracy.append(accs)
    print(confusion_matrix(Y_test,preds.round()),'Nodes: ', node,accs)
# print(accuracy)
sns.lineplot(x=nodes,y=accuracy)


# In[37]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error
nodes = [10,15,20,25,50,100,200]
accuracy =[]
for node in nodes:    
    model = RandomForestClassifier(criterion = 'entropy', random_state=0, n_estimators=node)
    model.fit(X_train,Y_train.values.ravel())
    preds = model.predict(X_test)
    accs = accuracy_score(Y_test,preds.round())
    accuracy.append(accs)
    print(confusion_matrix(Y_test,preds.round()),'No of Estimators: ', node,accs)
sns.lineplot(x=nodes,y=accuracy)
pickle.dump(model, open('RandomForestmodel.pkl','wb'))



# In[38]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error
nodes = [5,10,15,20,25,50,100]
accuracy =[]
for node in nodes:    
    classifier = KNeighborsClassifier(n_neighbors = node, metric = 'minkowski',p=2)
    classifier.fit(X_train,Y_train.values.ravel())
    preds = classifier.predict(X_test)
    accs = accuracy_score(Y_test,preds.round())
    accuracy.append(accs)
    print(confusion_matrix(Y_test,preds),'No of Neighbors: ', node,accs)

sns.lineplot(x=nodes,y=accuracy)


# In[39]:


import tensorflow as tf

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(loss = 'mean_squared_error' , optimizer = 'adam')
ann.fit(X_train.values, Y_train.values, batch_size = 32, epochs = 100)
y_pred = ann.predict(X_test.values)
accs = accuracy_score(Y_test,y_pred.round())
print(confusion_matrix(Y_test,y_pred.round()),accs)


# In[73]:


from xgboost import XGBClassifier   #XGBoostClassifier
classifier = XGBClassifier(max_depth=6)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)
accuracy_score(Y_test, y_pred)



# In[ ]:




