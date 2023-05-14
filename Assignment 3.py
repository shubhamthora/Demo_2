#!/usr/bin/env python
# coding: utf-8

# In[113]:


#Consider the diabetes dataset available on kaggle 
#(https://www.kaggle.com/code/milanvaddoriya/grid-search-cv-diabetes-dataset/data)
#Columns details are as follow:
#Pregnancies: Number of times pregnant
#Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#BloodPressure: Diastolic blood pressure (mm Hg)
#SkinThickness: Triceps skin fold thickness (mm)
#Insulin: 2-Hour serum insulin (mu U/ml)
#BMI: Body mass index (weight in kg/(height in m)^2)
#DiabetesPedigreeFunction: Diabetes pedigree function
#Age: Age (years)
#Outcome: Class variable (0 or 1)
#1:Explore dataset features impact on outcome class.
#2.If missing values are there,replace that with consideration of outcome class values.
#3.Build a best model among all classifier with optimum values of hyper-parameter.
#4.You may use gridsearchCV for the same.


# In[114]:


import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[115]:


#DATA Collrection And Analysis


# In[116]:


diabetes_dataset =pd.read_csv('diabetes (1).csv')


# In[117]:


diabetes_dataset


# In[118]:


diabetes_dataset.head()


# In[119]:


diabetes_dataset.shape


# In[120]:


diabetes_dataset.describe()


# In[121]:


diabetes_dataset['Outcome'].value_counts()


# In[122]:


#0 is represent Diabetic
#1 is non Diabetic


# In[123]:


diabetes_dataset.groupby('Outcome').mean()


# In[124]:


X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']  


# In[125]:


print(X)


# In[126]:


print(Y)


# In[127]:


#Data Std 


# In[128]:


scaler=StandardScaler()


# In[129]:


scaler.fit(X)


# In[130]:


standardized_data=scaler.transform(X)


# In[131]:


print(standardized_data)


# In[132]:


X=standardized_data
Y=diabetes_dataset['Outcome']


# In[133]:


print(X)
print(Y)


# In[134]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[135]:


print(X.shape,X_train.shape,X_test.shape)


# In[136]:


#Training model


# In[137]:


classifier =svm.SVC(kernel='linear')


# In[138]:


#training the support vector Machine Classifier


# In[139]:


classifier.fit(X_train,Y_train)


# In[140]:


#model evaluation


# In[141]:


#Accuracy Score on the training data


# In[142]:


X_train_prediction =classifier.predict(X_train)


# In[143]:


training_data_accuracy= accuracy_score(X_train_prediction,Y_train)


# In[144]:


print('Accuracy score of the training data:',training_data_accuracy)


# In[145]:


X_test_prediction =classifier.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction,Y_test)


# In[146]:


print('Accuracy score of the test data:',test_data_accuracy)


# In[147]:


#Making a Predictive System


# In[160]:


input_data=(1,85,66,29,0,26.6,0.351,31)


# In[161]:


#changing the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)


# In[162]:


input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


# In[163]:


#std the input_data
std_data =scaler.transform(input_data_reshaped)
print(std_data)


# In[164]:


prediction =classifier.predict(std_data)


# In[165]:


print(prediction)


# In[166]:


if (prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')


# In[ ]:




