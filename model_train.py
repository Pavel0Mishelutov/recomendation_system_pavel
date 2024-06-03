#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

    
def load_models():
    model_path = get_model_path("/my/super/path")
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 20000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

df = pd.read_sql('SELECT * FROM a_vahterkina_features_lesson_22_random_1000_users_data', con=SQLALCHEMY_DATABASE_URL)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month'] = df['timestamp'].apply(lambda x: x.month)
df['day'] = df['timestamp'].apply(lambda x: x.day)
df['hour'] = df['timestamp'].apply(lambda x: x.hour)
df['exp_group'] = df.exp_group.apply(lambda x: str(x))

df = df.sort_values(['month', 'day'],ascending = [True, True])
df = df[['gender','post_id','exp_group','topic',  'age', 'country','target']]

train_new= df.iloc[:df.shape[0]*8//10].copy()
test_new = df.iloc[df.shape[0]*8//10:].copy()

X_train = train_new.drop('target', axis=1)[['gender','post_id','exp_group','topic', 'age', 'country']]
X_test = test_new.drop('target', axis=1)[['gender','post_id','exp_group','topic', 'age', 'country']]

y_train = train_new['target']
y_test = test_new['target']


# In[2]:


boost = CatBoostClassifier(random_state=1,verbose=False)

boost.fit(X_train, y_train, cat_features=['gender','exp_group','topic','country'])


filename = 'sklearn_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(boost, file)

with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)


predict_train = loaded_model.predict(X_train)
predict_proba_train = loaded_model.predict_proba(X_train)


# In[3]:


filename = 'sklearn_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(boost, file)

