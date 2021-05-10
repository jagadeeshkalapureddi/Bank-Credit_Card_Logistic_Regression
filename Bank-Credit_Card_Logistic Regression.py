# # -----------------------------@ BANK CREDIT CARD ANALYSIS @--------------------------------------
# # --------------------------------! LOG REGRESSION !----------------------------------------------

# # IMPORT THE REQUIRED PACKAGES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# ### READ THE DATASET

df = pd.read_csv('BankCreditCard.csv')

# ### DATA UNDERSTANDING

# Check for the Data 
df.head()

# Check for the Dimensions.
df.shape

# Check for the column names and count.
print('Total_columns :', len(df.columns))
df.columns

# Check for the Summary of the data.
df.describe()

# Check for the Null values presence for each variable.
df.isna().sum()

# ## EXPLORATORY DATA ANALYSIS (EDA)

# Drop Customer ID because it won't reflect any impact on Y(Response Variable)
df.drop('Customer ID', axis = 1, inplace = True)

# Again check the columns for confirmation.
print('Total_Columns: ', len(df.columns))
df.columns

# ### DATA SPLITING INTO X AND Y

# Check for the Shape and value counts of x and y after splitting the data variables.
y = df.iloc[:, 23]
print('Dependent_Variable: ','\n','Dimensions: ', y.shape,'\n','Column_name: ', y.name, '\n')
print('Value_counts: ', '\n',y.value_counts(normalize = True).mul(100).round(1).astype(str) + '%', '\n')
x = df.iloc[:,0:23]
print('Independent_Variable: ','\n','Dimensions: ', x.shape,'\n','Column_names: ', x.columns, '\n')

# ## Data spliting into numerical and categorical
# 
# ### Numerical :
# Check for the Data set Information of each variable with data type.

df.info()

# #### In this all the variables are float and int type.

# Lets take individually based on their data types.
Num_type = ['float32', 'float64', 'int16', 'int32', 'int64']
x_num = x.select_dtypes(Num_type)
print('Total: ', len(x_num.columns))
x_num.columns

# ### Categorical :  (particularly select the binary categorical variables and remove them from Num_type)

x_cat = ['Gender', 'Academic_Qualification','Marital','Repayment_Status_Jan', 'Repayment_Status_Feb', 'Repayment_Status_March',
         'Repayment_Status_April', 'Repayment_Status_May','Repayment_Status_June']

# Drop the categorical variables from the numerical variables.
x_num.drop(x_cat, axis = 1, inplace = True)

# Select the categorical variables from the x dataset which is assigned above.
x_cat = x[x_cat]

# Categorical variables
print('Categorical variable_count: ', len(x_cat.columns))
x_cat.columns

# Numerical variables.
print('Numerical variable_count: ', len(x_num.columns))
x_num.columns

# ### EDA for Numerical Variables.

x_num.head()

x_num.describe()

x_num.isna().sum()

# ### OUTLIERS ANALYSIS.

plt.subplot(7,2,1)
sns.boxplot(x_num.iloc[:,0], color = 'blue')
plt.xlabel(x_num.iloc[:,0].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,0].name, fontsize = 15)

plt.subplot(7,2,2)
sns.boxplot(x_num.iloc[:,1], color = 'blue')
plt.xlabel(x_num.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,1].name, fontsize = 15)

plt.subplot(7,2,3)
sns.boxplot(x_num.iloc[:,2], color = 'blue')
plt.xlabel(x_num.iloc[:,2].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,2].name, fontsize = 15)

plt.subplot(7,2,4)
sns.boxplot(x_num.iloc[:,3], color = 'blue')
plt.xlabel(x_num.iloc[:,3].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,3].name, fontsize = 15)

plt.subplot(7,2,5)
sns.boxplot(x_num.iloc[:,4], color = 'blue')
plt.xlabel(x_num.iloc[:,4].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,4].name, fontsize = 15)

plt.subplot(7,2,6)
sns.boxplot(x_num.iloc[:,5], color = 'blue')
plt.xlabel(x_num.iloc[:,5].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,5].name, fontsize = 15)

plt.subplot(7,2,7)
sns.boxplot(x_num.iloc[:,6], color = 'blue')
plt.xlabel(x_num.iloc[:,6].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,6].name, fontsize = 15)

plt.subplot(7,2,8)
sns.boxplot(x_num.iloc[:,7], color = 'blue')
plt.xlabel(x_num.iloc[:,7].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,7].name, fontsize = 15)

plt.subplot(7,2,9)
sns.boxplot(x_num.iloc[:,8], color = 'blue')
plt.xlabel(x_num.iloc[:,8].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,8].name, fontsize = 15)

plt.subplot(7,2,10)
sns.boxplot(x_num.iloc[:,9], color = 'blue')
plt.xlabel(x_num.iloc[:,9].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,9].name, fontsize = 15)

plt.subplot(7,2,11)
sns.boxplot(x_num.iloc[:,10], color = 'blue')
plt.xlabel(x_num.iloc[:,10].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,10].name, fontsize = 15)

plt.subplot(7,2,12)
sns.boxplot(x_num.iloc[:,11], color = 'blue')
plt.xlabel(x_num.iloc[:,11].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,11].name, fontsize = 15)

plt.subplot(7,2,13)
sns.boxplot(x_num.iloc[:,12], color = 'blue')
plt.xlabel(x_num.iloc[:,12].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,12].name, fontsize = 15)

plt.subplot(7,2,14)
sns.boxplot(x_num.iloc[:,13], color = 'blue')
plt.xlabel(x_num.iloc[:,13].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,13].name, fontsize = 15)

plt.subplots_adjust(left=0, bottom=0, right=3, top=10)


print('Skewness: ')
print(x_num.iloc[:,0].name,":", x_num.iloc[:,0].skew())
print(x_num.iloc[:,1].name,":", x_num.iloc[:,1].skew())
print(x_num.iloc[:,2].name,":", x_num.iloc[:,2].skew())
print(x_num.iloc[:,3].name,":", x_num.iloc[:,3].skew())
print(x_num.iloc[:,4].name,":", x_num.iloc[:,4].skew())
print(x_num.iloc[:,5].name,":", x_num.iloc[:,5].skew())
print(x_num.iloc[:,6].name,":", x_num.iloc[:,6].skew())
print(x_num.iloc[:,7].name,":", x_num.iloc[:,7].skew())
print(x_num.iloc[:,8].name,":", x_num.iloc[:,8].skew())
print(x_num.iloc[:,9].name,":", x_num.iloc[:,9].skew())
print(x_num.iloc[:,10].name,":", x_num.iloc[:,10].skew())
print(x_num.iloc[:,11].name,":", x_num.iloc[:,11].skew())
print(x_num.iloc[:,12].name,":", x_num.iloc[:,12].skew())
print(x_num.iloc[:,13].name,":", x_num.iloc[:,13].skew())


# #### Ouliers were not treated because those are not Outliers.

# In[ ]:


plt.figure(figsize = (12,10))
sns.heatmap(x_num.corr(),annot = True)
plt.subplots_adjust(left=0.1, bottom=0.1, right=1, top=1)


# In[ ]:


# Mean-Max Normalization
x_num = (x_num - x_num.min()) / (x_num.max() - x_num.min())


# In[ ]:


x_num.head()


# ### EDA for Categorical Variables :

# In[ ]:


# Barplots for categorical variables with Response Variable.
plt.subplot(2,3,1)
sns.barplot(y = 'Credit_Amount', x = 'Default_Payment', hue = 'Gender', data = df)
plt.subplot(2,3,2)
sns.barplot(y = 'Age_Years', x = 'Default_Payment', hue = 'Gender', data = df)
plt.subplot(2,3,3)
sns.barplot(x = 'Gender', y = 'Default_Payment', hue = 'Marital', data = df)
plt.subplot(2,3,4)
sns.barplot(x = 'Marital', y = 'Default_Payment', data = df)
plt.subplot(2,3,5)
sns.barplot(x = 'Academic_Qualification', y = 'Default_Payment', hue = 'Marital', data = df)
plt.subplot(2,3,6)
sns.barplot(x = 'Academic_Qualification', y = 'Default_Payment', hue = 'Gender', data = df)

plt.subplots_adjust(left=0, bottom=0, right=2, top=2)


# In[ ]:


x_cat.head()


# In[ ]:


x_cat.isna().sum()


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

x_cat.Gender.replace([1, 2],['male', 'female'], inplace=True)

x_cat.Repayment_Status_Jan.replace([0,1,2,3,4,5,6],['janPaid','janDelay One','janDelay Two','janDelay Three','janDelay Four','janDelay Five','janDelay Six'], inplace = True)
x_cat.Repayment_Status_Feb.replace([0,1,2,3,4,5,6],['febPaid','febDelay One','febDelay Two','febDelay Three','febDelay Four','febDelay Five','febDelay Six'], inplace = True)
x_cat.Repayment_Status_March.replace([0,1,2,3,4,5,6],['marPaid','marDelay One','marDelay Two','marDelay Three','marDelay Four','marDelay Five','marDelay Six'], inplace = True)
x_cat.Repayment_Status_April.replace([0,1,2,3,4,5,6],['aprilPaid','aprilDelay One','aprilDelay Two','aprilDelay Three','aprilDelay Four','aprilDelay Five','aprilDelay Six'], inplace = True)
x_cat.Repayment_Status_May.replace([0,1,2,3,4,5,6],['mayPaid','mayDelay One','mayDelay Two','mayDelay Three','mayDelay Four','mayDelay Five','mayDelay Six'], inplace = True)
x_cat.Repayment_Status_June.replace([0,1,2,3,4,5,6],['junePaid','juneDelay One','juneDelay Two','juneDelay Three','juneDelay Four','juneDelay Five','juneDelay Six'], inplace = True)

x_cat.Marital.replace([1, 2, 3],['Married', 'Single', 'Abs'], inplace=True)

x_cat.Academic_Qualification.replace([1,2,3,4,5,6],['Undergraduate', 'Graduate','Postgraduate','Professional','Others',"Abm"], inplace=True)


# In[ ]:


x_cat.head()


# ### Check for values count of each categorical variable.

# In[ ]:


print(x_cat.iloc[:,0].name," :")
print(x_cat.iloc[:,0].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,1].name," :")
print(x_cat.iloc[:,1].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,2].name," :")
print(x_cat.iloc[:,2].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,3].name," :")
print(x_cat.iloc[:,3].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,4].name," :")
print(x_cat.iloc[:,4].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,5].name," :")
print(x_cat.iloc[:,5].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,6].name," :")
print(x_cat.iloc[:,6].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,7].name," :")
print(x_cat.iloc[:,7].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
print('\n')
print(x_cat.iloc[:,8].name," :")
print(x_cat.iloc[:,8].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')


# ###  Barplots for each categorical Variable.

# In[ ]:


plt.subplot(5,2,1)
x_cat.iloc[:,0].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,0].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,0].name, fontsize = 15)

plt.subplot(5,2,2)
x_cat.iloc[:,1].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,1].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,1].name, fontsize = 15)

plt.subplot(5,2,3)
x_cat.iloc[:,2].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,2].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,2].name, fontsize = 15)

plt.subplot(5,2,4)
x_cat.iloc[:,3].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,3].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,3].name, fontsize = 15)

plt.subplot(5,2,5)
x_cat.iloc[:,4].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,4].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,4].name, fontsize = 15)

plt.subplot(5,2,6)
x_cat.iloc[:,5].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,5].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,5].name, fontsize = 15)

plt.subplot(5,2,7)
x_cat.iloc[:,6].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,6].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,6].name, fontsize = 15)

plt.subplot(5,2,8)
x_cat.iloc[:,7].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,7].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,7].name, fontsize = 15)

plt.subplot(5,2,9)
x_cat.iloc[:,8].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(x_cat.iloc[:,8].name, fontsize = 20)
plt.title('Barplot_ '+ x_cat.iloc[:,8].name, fontsize = 15)

plt.subplots_adjust(left=0, bottom=0.5, right=2, top=10)


# In[ ]:


x_cat.columns


# ### Dummification

# In[ ]:


x_cate = pd.get_dummies(x_cat, drop_first=True)
x_cate.head()


# ### Combine Categorical And Numerical Variables

# In[ ]:


print("Total Categorical variable:",len(x_cate.columns))
print("Total Numerical variable:",len(x_num.columns))
x_final = pd.concat([x_cate, x_num], axis = 1)
print("Final Dataframe shape:",x_final.shape)


# In[ ]:


x_final.head()


# In[ ]:


Final_data = pd.concat([y,x_final], sort='True', axis = 1)
Final_data.head()


# ### For Building Model:
# 
# **x_final** is the x variables.
# **y** is the y variable.
# 
# ### Split the data into x and y for training and testing :

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y, train_size = 0.8, random_state = 100)
print('x_train shape :', x_train.shape)
print('x_test shape :', x_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# ### Fit the train model and predict the test set.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm = LogisticRegression()
glm = glm.fit(x_train, y_train)
predicted = glm.predict(x_test)


# ## Model Evaluation :
# 
# ### Full Model:

# ### VIF 

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x_final.columns

vif_data["VIF"] = [variance_inflation_factor(x_final.values, i) 
                          for i in range(len(x_final.columns))] 
  
print(vif_data)


# ### LOGIT OLS SUMMARY

# In[ ]:


import statsmodels.api as sm
x_train_sm = sm.Logit(y_train, x_train)
lm = x_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm.summary())


# ### Fit the train model and predict the test set

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm = LogisticRegression()
glm = glm.fit(x_train, y_train)
predicted = glm.predict(x_test)


# ### RFE

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import RFE
rfe = RFE(glm, 10)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_)


# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# In[ ]:


# Function For Logistic Regression Create Summary For Logistic Regression

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def get_summary(y_test,predicted):
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test,predicted)
    TP = conf_mat[0,0:1]
    FP = conf_mat[0,1:2]
    FN = conf_mat[1,0:1]
    TN = conf_mat[1,1:2]
    
    accuracy = (TP+TN)/((FN+FP)+(TP+TN))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall =  TP / (TP + FN)
    fScore = (2 * recall * precision) / (recall + precision)
    auc = roc_auc_score(y_test, predicted)

    print("Confusion Matrix:\n",conf_mat)
    print("Accuracy:",accuracy)
    print("Sensitivity :",sensitivity)
    print("Specificity :",specificity)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-score:",fScore)
    print("AUC:",auc)
    print('\n')
    print("ROC curve:")
    fpr, tpr, thresholds = roc_curve(y_test, predicted)
    plot_roc_curve(fpr, tpr)


# In[ ]:


get_summary(y_test,predicted)


# ### Model-1 (Backward Elimination) -- Top 10 from RFE

# ### Train and Test the data.

# In[ ]:


x1_train = x_train[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Marital_Abs', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan', 'Previous_Payment_Feb',
       'Previous_Payment_April',
       'Previous_Payment_May', 'Previous_Payment_June']]

x1_test = x_test[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Marital_Abs', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan', 'Previous_Payment_Feb',
       'Previous_Payment_April',
       'Previous_Payment_May', 'Previous_Payment_June']]


# ### VIF

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x1_train.columns

vif_data["VIF"] = [variance_inflation_factor(x1_train.values, i) 
                          for i in range(len(x1_train.columns))] 
  
print(vif_data)


# ### LOGIT OLS SUMMARY

# In[ ]:


import statsmodels.api as sm
x1_train_sm = sm.Logit(y_train, x1_train)
lm1 = x1_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm1.summary())


# ### Fit the train model and predict the test set

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm1 = LogisticRegression()
glm1 = glm1.fit(x1_train, y_train)
predicted1 = glm1.predict(x1_test)


# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

get_summary(y_test,predicted1)


# ### Model-2 (Backward Elimination) -- Marital-Abs

# ### Train and Test the data.

x2_train = x_train[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan', 'Previous_Payment_Feb',
       'Previous_Payment_April',
       'Previous_Payment_May', 'Previous_Payment_June']]

x2_test = x_test[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan', 'Previous_Payment_Feb',
       'Previous_Payment_April',
       'Previous_Payment_May', 'Previous_Payment_June']]


# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x2_train.columns

vif_data["VIF"] = [variance_inflation_factor(x2_train.values, i) 
                          for i in range(len(x2_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x2_train_sm = sm.Logit(y_train, x2_train)
lm2 = x2_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm2.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm2 = LogisticRegression()
glm2 = glm2.fit(x2_train, y_train)
predicted2 = glm2.predict(x2_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

get_summary(y_test,predicted2)

# ### Model-3 (Backward Elimination) -- Pre_Pay_june

# ### Train and Test the data.

x3_train = x_train[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan', 'Previous_Payment_Feb',
       'Previous_Payment_April',
       'Previous_Payment_May']]

x3_test = x_test[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan', 'Previous_Payment_Feb',
       'Previous_Payment_April',
       'Previous_Payment_May']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x3_train.columns

vif_data["VIF"] = [variance_inflation_factor(x3_train.values, i) 
                          for i in range(len(x3_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x3_train_sm = sm.Logit(y_train, x3_train)
lm3 = x3_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm3.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm3 = LogisticRegression()
glm3 = glm3.fit(x3_train, y_train)
predicted3 = glm3.predict(x3_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

get_summary(y_test,predicted3)

# ### Model-4 (Backward Elimination) -- Pre_Pay_Apr, Feb

# ### Train and Test the data.

x4_train = x_train[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan',
       'Previous_Payment_May']]

x4_test = x_test[['Academic_Qualification_Others', 'Academic_Qualification_Professional', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Feb_Bill_Amount', 'Previous_Payment_Jan',
       'Previous_Payment_May']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x4_train.columns

vif_data["VIF"] = [variance_inflation_factor(x4_train.values, i) 
                          for i in range(len(x4_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x4_train_sm = sm.Logit(y_train, x4_train)
lm4 = x4_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm4.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm4 = LogisticRegression()
glm4 = glm4.fit(x4_train, y_train)
predicted4 = glm4.predict(x4_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

get_summary(y_test,predicted4)

# ### Model-5 (Backward Elimination) -- Aca-prof, feb-bill-amount

# ### Train and Test the data.

x5_train = x_train[['Academic_Qualification_Others', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Previous_Payment_Jan',
       'Previous_Payment_May']]

x5_test = x_test[['Academic_Qualification_Others', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid',
       'Credit_Amount', 'Age_Years', 'Previous_Payment_Jan',
       'Previous_Payment_May']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x5_train.columns

vif_data["VIF"] = [variance_inflation_factor(x5_train.values, i) 
                          for i in range(len(x5_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x5_train_sm = sm.Logit(y_train, x5_train)
lm5 = x5_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm5.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm5 = LogisticRegression()
glm5 = glm5.fit(x5_train, y_train)
predicted5 = glm5.predict(x5_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

get_summary(y_test,predicted5)

# ### Model-6 (Backward Elimination) -- payment-may

# ### Train and Test the data.

x6_train = x_train[['Academic_Qualification_Others', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid', 'Credit_Amount', 'Age_Years', 'Previous_Payment_Jan',]]

x6_test = x_test[['Academic_Qualification_Others', 'Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid', 'Credit_Amount', 'Age_Years', 'Previous_Payment_Jan',]]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x6_train.columns

vif_data["VIF"] = [variance_inflation_factor(x6_train.values, i) 
                          for i in range(len(x6_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x6_train_sm = sm.Logit(y_train, x6_train)
lm6 = x6_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm6.summary())

# ### Fit the train model and predict the test set.

from sklearn.linear_model import LogisticRegression
glm6 = LogisticRegression()
glm6 = glm6.fit(x6_train, y_train)
predicted6 = glm6.predict(x6_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

get_summary(y_test,predicted6)

# ### Model-7 (Backward Elimination) -- Aca-Qua-others

# ### Train and Test the data.

x7_train = x_train[['Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid', 'Credit_Amount', 'Age_Years', 'Previous_Payment_Jan']]

x7_test = x_test[['Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid', 'Credit_Amount', 'Age_Years', 'Previous_Payment_Jan']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x7_train.columns

vif_data["VIF"] = [variance_inflation_factor(x7_train.values, i) 
                          for i in range(len(x7_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x7_train_sm = sm.Logit(y_train, x7_train)
lm7 = x7_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm7.summary())

# ### Fit the train model and predict the test set.

from sklearn.linear_model import LogisticRegression
glm7 = LogisticRegression()
glm7 = glm7.fit(x7_train, y_train)
predicted7 = glm7.predict(x7_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

get_summary(y_test,predicted7)

# ### Model-8 (Backward Elimination) -- remains CA, Age_years

# ### Train and Test the data.

x8_train = x_train[['Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid', 'Credit_Amount', 'Age_Years']]

x8_test = x_test[['Repayment_Status_Jan_janDelay One',
       'Repayment_Status_Jan_janPaid', 'Credit_Amount', 'Age_Years']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x8_train.columns

vif_data["VIF"] = [variance_inflation_factor(x8_train.values, i) 
                          for i in range(len(x8_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x8_train_sm = sm.Logit(y_train, x8_train)
lm8 = x8_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm8.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm8 = LogisticRegression()
glm8 = glm8.fit(x8_train, y_train)
predicted8 = glm8.predict(x8_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# ## Final Selection:

conf_mat1 = confusion_matrix(y_test,predicted1)
TP1 = conf_mat1[0,0:1]
FP1 = conf_mat1[0,1:2]
FN1 = conf_mat1[1,0:1]
TN1 = conf_mat1[1,1:2]
accuracy1 = (TP1+TN1)/((FN1+FP1)+(TP1+TN1))

conf_mat2 = confusion_matrix(y_test,predicted2)
TP2 = conf_mat2[0,0:1]
FP2 = conf_mat2[0,1:2]
FN2 = conf_mat2[1,0:1]
TN2 = conf_mat2[1,1:2]
accuracy2 = (TP2+TN2)/((FN2+FP2)+(TP2+TN2))

conf_mat3 = confusion_matrix(y_test,predicted3)
TP3 = conf_mat3[0,0:1]
FP3 = conf_mat3[0,1:2]
FN3 = conf_mat3[1,0:1]
TN3 = conf_mat3[1,1:2]
accuracy3 = (TP3+TN3)/((FN3+FP3)+(TP3+TN3))

conf_mat4 = confusion_matrix(y_test,predicted4)
TP4 = conf_mat4[0,0:1]
FP4 = conf_mat4[0,1:2]
FN4 = conf_mat4[1,0:1]
TN4 = conf_mat4[1,1:2]
accuracy4 = (TP4+TN4)/((FN4+FP4)+(TP4+TN4))

conf_mat5 = confusion_matrix(y_test,predicted5)
TP5 = conf_mat5[0,0:1]
FP5 = conf_mat5[0,1:2]
FN5 = conf_mat5[1,0:1]
TN5 = conf_mat5[1,1:2]
accuracy5 = (TP5+TN5)/((FN5+FP5)+(TP5+TN5))

conf_mat6 = confusion_matrix(y_test,predicted6)
TP6 = conf_mat6[0,0:1]
FP6 = conf_mat6[0,1:2]
FN6 = conf_mat6[1,0:1]
TN6 = conf_mat6[1,1:2]
accuracy6 = (TP6+TN6)/((FN6+FP6)+(TP6+TN6))

conf_mat7 = confusion_matrix(y_test,predicted7)
TP7 = conf_mat7[0,0:1]
FP7 = conf_mat7[0,1:2]
FN7 = conf_mat7[1,0:1]
TN7 = conf_mat7[1,1:2]
accuracy7 = (TP7+TN7)/((FN7+FP7)+(TP7+TN7))

conf_mat8 = confusion_matrix(y_test,predicted8)
TP8 = conf_mat8[0,0:1]
FP8 = conf_mat8[0,1:2]
FN8 = conf_mat8[1,0:1]
TN8 = conf_mat8[1,1:2]
accuracy8 = (TP8+TN8)/((FN8+FP8)+(TP8+TN8))

# ### Print all accuracies for best selection.

print('accuracy1 : ', accuracy1)
print('\n')
print('Model-1 : ', x1_test.columns)
print('Variable_Count-1 : ', len(x1_test.columns))

print('\n')
print('accuracy2 : ', accuracy2)
print('\n')
print('Model-2 : ', x2_test.columns)
print('Variable_Count-2 : ', len(x2_test.columns))

print('\n')
print('accuracy3 : ', accuracy3)
print('\n')
print('Model-3 : ', x3_test.columns)
print('Variable_Count-3 : ', len(x3_test.columns))

print('\n')
print('accuracy4 : ', accuracy4)
print('\n')
print('Model-4 : ', x4_test.columns)
print('Variable_Count-4 : ', len(x4_test.columns))

print('\n')
print('accuracy5 : ', accuracy5)
print('\n')
print('Model-5 : ', x5_test.columns)
print('Variable_Count-5 : ', len(x5_test.columns))

print('\n')
print('accuracy6 : ', accuracy6)
print('\n')
print('Model-6 : ', x6_test.columns)
print('Variable_Count-6 : ', len(x6_test.columns))

print('\n')
print('accuracy7 : ', accuracy7)
print('\n')
print('Model-7 : ', x7_test.columns)
print('Variable_Count-7 : ', len(x7_test.columns))

print('\n')
print('accuracy8 : ', accuracy8)
print('\n')
print('Model-8 : ', x8_test.columns)
print('Variable_Count-8 : ', len(x8_test.columns))

# ### Final Model Summary Details of Tested and predicted data.

print(get_summary(y_test,predicted8))

# ### Finally we got the result :  Model-8
# 
# #### In building the model I just taken the top 10 variable from RFE Method.
# 
# #### As per the previous model seen  AUC 81 % that the model get fitted with 'Repayment_Status_Jan_janDelay One', 
# 'Repayment_Status_Jan_janPaid', 'Credit_Amount', 'Age_Years'

# --------------------------------------------------------------------------------------------------------------------**@** **Analysis done by,**
# ----------------------------------------------------------------------------------------------------------------------- **@**   **Jagadeesh K**
