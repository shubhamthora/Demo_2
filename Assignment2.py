#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Problem Statement-
#Consider a dataset provided by   (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29) 
#and it contains the following attribute information(in order):Sample code 
#    number: id 
#    Uniformity of Cell Size: 1–10
#    Uniformity of Cell Shape: 1–10
 #   Marginal Adhesion: 1–10 hi
 #   Single Epithelial Cell Size: 1–10
 #   Bare Nuclei: 1–10
#    Bland Chromatin: 1–10
#    Normal Nucleoli: 1–10
#    Mitoses: 1–10
#    Class: (2 for benign, 4 for malignant)
#
#Build a logistic model  with best accuracy.It's good if you implement it from scratch.
#Also find the optimum value of threshold  for you model.
#Explore hyperparameter tuning using cross-validation.


# In[2]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
import sklearn.datasets  
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


#  Data Collection & Processing


# In[4]:


breast_cancer_dataset =sklearn.datasets.load_breast_cancer()


# In[5]:


print(breast_cancer_dataset)


# In[6]:


data_frame=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)


# In[7]:


data_frame.head() 


# In[8]:


#adding target column to the data frame
data_frame['label'] =breast_cancer_dataset.target


# In[9]:


data_frame.tail() 


# In[10]:


data_frame.shape


# In[11]:


data_frame.info()


# In[12]:


data_frame.isnull().sum()


# In[13]:


#statistics
data_frame.describe()


# In[14]:


#checking the distribution
data_frame['label'].value_counts()


# In[15]:


data_frame.groupby('label').mean()


# In[16]:


#seperating feature & Target
X=data_frame.drop(columns='label',axis=1)
Y=data_frame['label']


# In[17]:


print(X)


# In[18]:


print(Y)


# In[19]:


#splitting data into training &testing data


# In[20]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=2)


# In[21]:


print(X.shape,X_train.shape,X_test.shape)


# In[22]:


#model Training
  #logistic regression


# model =Logisticregession()

# In[23]:


model =LogisticRegression()


# In[24]:


model.fit(X_train,Y_train)


# In[25]:


#evalution


# In[26]:


X_train_prediction= model.predict(X_train)


# In[27]:


training_data_accuracy=accuracy_score(Y_train,X_train_prediction)


# In[28]:


print('Accuracy Training Data=',training_data_accuracy)


# In[29]:


X_test_prediction= model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)


# In[30]:


print('Accuracy Training Data=',test_data_accuracy)


# In[31]:


#Bulding a predictive System


# In[32]:


input_data=(19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.00615,0.04006,0.03832,0.02058,0.0225,0.004571,23.57,25.53,152.5,1709,0.1444,0.4245,0.4504,0.243,0.3613,0.08758
)
           


# In[33]:


input_data_asnumpy_array=np.asarray(input_data)


# In[34]:


input_data_reshape=input_data_asnumpy_array.reshape(1,-1)


# In[35]:


prediction =model.predict(input_data_reshape)


# In[36]:


print(prediction)


# In[37]:


if(prediction[0]==0):
    print("The Breast cancer is malignent")
else:
    print("The Breast Cancer is benign")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




