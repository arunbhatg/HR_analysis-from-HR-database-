#!/usr/bin/env python
# coding: utf-8

# In[99]:



import sklearn
import numpy as np
import matplotlib as mp
import pandas as pd


# In[100]:


import mysql.connector

db = mysql.connector.connect(
   host="localhost",
   user="root",
   passwd="1234"
   
)


# In[101]:


import mysql.connector as sql
import pandas as pd

db_connection = sql.connect(host='localhost', database='arun', user='root', password='1234')
db_cursor = db_connection.cursor()
db_cursor.execute('SELECT * FROM hr_comma_sep')

table_rows = db_cursor.fetchall()


df = pd.DataFrame(table_rows, columns=db_cursor.column_names)


# In[102]:


df.head()


# In[103]:


df.groupby('left').describe()


# In[104]:


df.corr()


# In[105]:


sd = pd.get_dummies(df['salary'])
dn = pd.concat([df, sd], axis=1)


# In[106]:


dn.drop('salary', axis=1, inplace=True)


# In[107]:


dn.head()


# In[108]:


X = dn.drop(['sales', 'left', 'high'], axis=1)
y = dn['left']


# In[109]:


import sklearn
from sklearn.model_selection import train_test_split


# In[110]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:





# In[111]:


from sklearn.ensemble import RandomForestClassifier


# In[112]:


m = RandomForestClassifier(n_estimators=100)


# In[113]:


m.fit(X_train, y_train)


# In[114]:


p = m.predict(X_test)


# In[115]:


from sklearn.metrics import confusion_matrix, classification_report


# In[116]:


print(confusion_matrix(y_test, p))
print('\n')
print(classification_report(y_test, p))


# In[117]:


rfc = m.score(X_train, y_train)
print('RFC Train Score:',rfc)
rfctst = m.score(X_test, y_test)
print('RFC Test Score:',rfctst)


# In[118]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)


# In[ ]:





# In[119]:


from sklearn.neighbors import KNeighborsClassifier


# In[120]:


knn = KNeighborsClassifier(n_neighbors=77)


# In[121]:


knn.fit(X_train, y_train)


# In[122]:


kp = knn.predict(X_test)


# In[123]:


print(confusion_matrix(y_test, kp))


# In[124]:


print(classification_report(y_test, kp))


# In[125]:



from pandas.io import sql
from sqlalchemy import create_engine


# In[126]:


result= pd.DataFrame(y_test)
result


# In[127]:


result['predicted']=kp


# In[130]:


result


# In[ ]:





# In[132]:


from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:1234@localhost:3306/arun')

result.to_sql(name='ab', con=engine,index='id',if_exists = 'replace')


# In[92]:





# In[ ]:




