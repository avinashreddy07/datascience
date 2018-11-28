
# coding: utf-8

# In[1]:


# Importing the required libraries
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb


# In[2]:


os.getcwd()
os.chdir("E:\DataScience\Capstone_Proj")


# In[3]:


# Importing the dataset
df = pd.read_csv("Purchase.csv")


# In[4]:


df.head(10)
#df.dtypes


# In[5]:


# Extracting day of week, month and year from 'Original_Quote_Date' using datetime function from pandas
df['Date'] = pd.to_datetime(pd.Series(df['Original_Quote_Date']))
df = df.drop(["Original_Quote_Date","QuoteNumber"], axis=1)
df['weekday'] = df['Date'].dt.dayofweek
df['weekyear'] = df['Date'].dt.weekofyear
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month


# In[7]:


#df.isnull().count()
#df.isna().count()
#df.isnull().sum()
#df = df.drop(["weekyear","Year"], axis=1)
#df.describe()
#sum(df.isnull().values.ravel())


# In[8]:


df = df.drop(["Date"], axis=1)


# In[9]:


df.isnull().values.ravel().sum()


# In[10]:


df = df.fillna(-99) # Filling missing values with some random number


# In[11]:


# Converting the object columns to numeric using LableEncoder
for f in df.columns:
    if df[f].dtype == 'object':
        #print(f)
        lb1 = LabelEncoder()
        lb1.fit(list(df[f].values))
        df[f] = lb1.transform(list(df[f].values))


# In[12]:


# Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X = df.drop(['QuoteConversion_Flag'], axis=1)
y = df['QuoteConversion_Flag']
#y.dtype

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.2)

print(X_train.shape)
print(y_train.shape)


# In[13]:


#X_train


# # Xgboost     https://xgboost.readthedocs.io/en/latest/parameter.html

# In[14]:


# Fitting XGBoost to the Training set
clf = xgb.XGBClassifier(objective= "binary:logistic",
                       eta = 0.1,
                       n_estimators = 100,
                       verbose =1,
                       silent = False)

xbg_mobel = clf.fit(X_train, y_train, eval_metric="error")


# In[15]:


clf


# In[22]:


# Predicting the test results and Making the confusion matrix
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

pred = xbg_mobel.predict(X_test)


# In[23]:


pred[1]


# In[24]:


#predTrain = xbg_mobel.predict(X_train)


# In[25]:


probs = xbg_mobel.predict_proba(X_test)


# In[26]:


print(classification_report(y_test,pred))


# In[21]:


# print(classification_report(y_train,predTrain)) ## no need


# In[27]:


print(accuracy_score(pred,y_test))


# In[28]:


# Trying to improve the accuracy by changing max_depth parameters
clf1 = xgb.XGBClassifier(objective= "binary:logistic",
                       eta = 0.1,
                       n_estimators = 100,max_depth=10,
                       verbose =10,
                       silent = 0)


# In[30]:


xbg_mobel = clf1.fit(X_train, y_train, eval_metric="error")


# In[31]:


pred = xbg_mobel.predict(X_test)


# In[32]:


probs = xbg_mobel.predict_proba(X_test)


# In[33]:


print(classification_report(y_test,pred))


# In[35]:


print(accuracy_score(pred,y_test))

################################################################################################################################
# # accuracy has been increased from 0.91 to 0.92 by increasing max_depth, following are various parameters to improve accuracy#
# # # best params:eta=0.1,n_estimators=100,max_depth=default(3),f1:0.74                                                        #
# # # best params:eta=0.1,n_estimators=100,max_depth=10,f1:0.77                                                                #
# # # bestparams:eta=0.1,n_estimators=100,max_depth=10,subsample=0.6,colsample_bytree=0.6,f1:0.78                              #
################################################################################################################################
# In[36]:


# Applying KFold Cross Validation
from sklearn.model_selection import StratifiedKFold,cross_val_score


# In[37]:


kfold = StratifiedKFold(n_splits=3, random_state=7)
result = cross_val_score(clf,X_train,y_train,cv=kfold)


# In[38]:


result
## we wiil get each fold acc


# In[39]:


result.std()


# In[40]:


result.mean()

