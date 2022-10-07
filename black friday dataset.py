#!/usr/bin/env python
# coding: utf-8

# ## black friday dataset eda and feature engineering
# 
# # cleaning and preparing the data for model training
# ## a company wants to understand the costumer purchase behaviour against various product

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#getting train dataset
df_train=pd.read_csv('train.csv')


# In[3]:


df_train.head()


# In[4]:


df_train.shape


# In[5]:


# getting test dataset
df_test=pd.read_csv('test.csv')


# In[6]:


df_test.head()


# In[7]:


# now we will merge both test and train data to create full dataset
df=df_train.append(df_test)
df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


#user id is not useful,axis=1 (columnwise remove)
df.drop(['User_ID'],axis=1,inplace=True)


# In[11]:


df


# In[12]:


df.isnull().sum()


# In[13]:


# to change categorical value into int type
#mapping or one hot encoder 
#one hot encoder we have to perform many other things like adding and dropping (pd.get_dummies)
#mapping
df['Gender']=df['Gender'].map({'F':0,'M':1})


# In[14]:


df.head()


# In[15]:


# handle categorical feature
df['Age'].unique()


# In[16]:


# or use label encoding
#df['Age']=df['Age'].map({'0-17':1,'8-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})
df_age=pd.get_dummies(df['Age'],drop_first=True)


# In[17]:


df_age.head()


# In[18]:


df=pd.concat([df,df_age],axis=1)
df.head()


# In[19]:


df.drop('Age',axis=1,inplace=True)


# In[20]:


df


# In[21]:


#for city category(using one hot encoding)
#drop_first is used as we have 3 category so only two can be sufficient 
df_city=pd.get_dummies(df['City_Category'],drop_first=True)


# In[22]:


df_city.head()


# In[23]:


df=pd.concat([df,df_city],axis=1)
df.head()


# In[24]:


df.drop('City_Category',axis=1,inplace=True)


# In[25]:


df


# In[26]:


df.info()


# In[27]:


#Stay_In_Current_City_Years categorical value
df_stay=pd.get_dummies(df['Stay_In_Current_City_Years'],drop_first=True)


# In[28]:


df=pd.concat([df,df_stay],axis=1)
df.head()


# In[29]:


df.drop('Stay_In_Current_City_Years',axis=1,inplace=True)


# In[30]:


df


# In[31]:


df.info()


# In[32]:


# missing values
df.isnull().sum()


# In[33]:


# replace the missing value with mode
df['Product_Category_2'].unique()


# In[34]:


df['Product_Category_2'].value_counts()


# In[35]:


# replacing by mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[36]:


df['Product_Category_2'].isnull().sum()


# In[37]:


# removing null for purchase3

df['Product_Category_3'].unique()


# In[38]:


df['Product_Category_3'].value_counts()


# In[39]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[40]:


df['Product_Category_3'].isnull().sum()


# In[41]:


df


# In[42]:


#purchase is test data so it will remail null only
df.isnull().sum()


# In[43]:


df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)


# In[44]:


df.info()


# In[99]:


##visualisation
## sns.pairplot(df)


# In[45]:


# visualisation of gender wrt purchase
# male purchase more 
sns.barplot('Gender','Purchase',hue='Gender',data=df)


# In[46]:


#purchase with occupation
sns.barplot('Occupation','Purchase',hue='Gender',data=df)


# In[47]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df)


# In[48]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df) 


# In[49]:


sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df)


# ## product type 1 is purchased more

# # Feature scaling

# In[50]:


df_test=df[df['Purchase'].isnull()]#test data is null data


# In[51]:


df_train=df[~df['Purchase'].isnull()]#train data is not null data


# In[52]:


X=df_train.drop('Purchase',axis=1)


# In[53]:


X.head()


# In[54]:


X.shape


# In[55]:


Y=df_train['Purchase']


# In[56]:


Y.shape


# In[57]:


Y


# In[58]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)


# In[59]:


X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)


# In[60]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(X_train)
x_test=sc.transform(X_test)


# In[ ]:




