#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from datetime import datetime
from typing import List
from fastapi import FastAPI , Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, TIMESTAMP, cast, func,ForeignKey
from sqlalchemy.orm import relationship , Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd


# In[5]:


SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
u = pd.read_sql("SELECT distinct user_id, gender, exp_group, age,country FROM public.user_data",con = SQLALCHEMY_DATABASE_URL)
p = pd.read_sql('SELECT distinct post_id, topic,text FROM public.post_text_df',con = SQLALCHEMY_DATABASE_URL )


# In[6]:


import pickle
with open('sklearn_model.pkl', 'rb') as f:
    boost_model = pickle.load(f)


# In[7]:


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


app = FastAPI()
def get_db():
    with SessionLocal() as db:
        return db

def top_posts(id, model, limit):

    p1 = p.copy()
    u1 = u[u.user_id == id].copy()
    
    a = pd.merge(u1, p1, how='cross')
    a1 = a[['gender','post_id','exp_group','topic', 'age', 'country']]
    a1['predict_proba'] = boost_model.predict_proba(a1)[:, 1]
    a1 = a1.sort_values(by='predict_proba',ascending = False).head(limit)
    a1 = pd.merge(a1['post_id'], p1, on = 'post_id', how='inner')[['post_id','text','topic']]
#     a = df[df.user_id == id][['post_id','text','topic']]
    posts = [PostGet(id=row["post_id"], text=row["text"], topic=row["topic"]) for _, row in a1.iterrows()]
    return posts


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
            id: int,
            time: datetime,
            limit: int = 5) -> List[PostGet]:
    
    posts = top_posts(id,boost_model, limit)
    return posts

