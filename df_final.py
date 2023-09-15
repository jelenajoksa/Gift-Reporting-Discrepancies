
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy
from spacy.lang.sl import Slovenian
import seaborn as sns
import seaborn.objects as so
import dtale
import statsmodels
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.graph_objs as go
from pandas_profiling import ProfileReport
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.io as pio
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from deep_translator import GoogleTranslator


df_en = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_en.csv')
df1 = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/gifts_en.csv')
df_all = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_en_all.csv')

#check if all the df are sorted equally
len(df_en)
len(df1)
len(df_all)

df_en.head()
df1.head()
df_all.head()
#############


df1['Prejemnik'].unique()
df1['Lastnik darila'].unique()

df_all['recipient'].unique()
df_all['gift_owner'].unique()

df_all.columns

df1[(df1['Prejemnik']=='Družinski član uradne osebe') | (df1['Prejemnik']=='družinski član funkcionarja') ]

df_all[(df_all['recipient']=='family member official') ]


df_all['gift_owner'] = df_all['gift_owner'].replace('official','functionary')
df_all['gift_owner'] = df_all['gift_owner'].replace('recipient','official person')

df_all.columns

df_all['occasion']




df_all['Date'] = pd.to_datetime(df_all['date_str'], format='%Y-%m-%d')

df_all.columns

#lower.case
textual_columns = ['recipient_org', 'recipient', 'donor', 'gift_type','delivery_type', 'Description', 'gift_owner','value_determination_method','occasion']
df_all[textual_columns] = df_all[textual_columns].apply(lambda x: x.str.lower())


# Remove stopwords from the textual columns
stop_words = set(stopwords.words('english'))  # Set the language of stopwords
df_all[textual_columns] = df_all[textual_columns].apply(lambda x: x.apply(lambda y: ' '.join([word for word in str(y).split() if word not in stop_words])))




df_all.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_final.csv', index=False, header=True)
