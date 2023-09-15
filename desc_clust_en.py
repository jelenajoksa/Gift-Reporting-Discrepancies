import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from spacy.lang.sl import Slovenian
import seaborn as sns
import seaborn.objects as so
import dtale
import pandas_profiling
import statsmodels
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.graph_objs as go
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.io as pio
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from deep_translator import GoogleTranslator



df_en = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_en.csv')
df2 = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/gifts_en.csv')

df_en['Description'] = df2['Opis_EN']

df_en.columns

df_en.rename(columns = {'Datum_EN':'date_str',
                        'Naziv_EN':'recipient_org',
                        'Prejemnik_EN':'recipient',
                        'Darovalec_EN':'donor',
                        'Tip darila_EN':'gift_type',
                        'Način izročitve_EN':'delivery_type'}, inplace = True)

df_en['gift_type'].unique()

df_en['gift_type'] = df_en['gift_type'].replace('casually','occasional')

#More translation

df_en['gift_owner'].unique()
df_en['Lastnik darila'] = df_en['Lastnik darila'].replace('Subjekta javnega sektorja (delodajalca)','public sector entity (employer)')
df_en['Lastnik darila'] = df_en['Lastnik darila'].replace('organizacija, v kateri funkcionar opravlja funkcijo','organization where the official holds a position')
df_en['Lastnik darila'] = df_en['Lastnik darila'].replace('država','country')
df_en['Lastnik darila'] = df_en['Lastnik darila'].replace('funkcionar','official')
df_en['Lastnik darila'] = df_en['Lastnik darila'].replace('lokalna skupnost','local community')
df_en['Lastnik darila'] = df_en['Lastnik darila'].replace('Prejemnika','recipient')
df_en['Lastnik darila'] = df_en['Lastnik darila'].replace('darilo je bilo vrnjeno obdarovalcu','gift was returned to the donor')

df_en.rename(columns = {'Lastnik darila':'gift_owner'}, inplace = True)

df2[df2['Lastnik darila'] == 'Prejemnika']
(df_en['Prejemnik']).unique()
(df_en['recipient']).unique()

def translate_text(text):
    if pd.isnull(text) or text == '':
        return ''

    try:
        return GoogleTranslator(source='sl', target='en').translate(text)
    except Exception as e:
        print(f"Error occurred during translation: {e}")
        return ''

len(df_en['Način določitve vrednosti'].unique())
len(df_en['Razlog za izročitev'].unique())

df_en['Način določitve vrednosti'][55]

df_en['value_determination_method'] = df_en['Način določitve vrednosti'].astype(str).apply(translate_text)

df_en['occasion'] = df_en['Razlog za izročitev'].astype(str).apply(translate_text)

df_en.columns
df_en.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_en_all.csv', index=False, header=True)

#EDA

#Gift Donor


###############DAROVALEC

df_en = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/df_en_all.csv')

df = df_en

# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['donor'])]['donor']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:4]))

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift donor'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}, 'showgrid': True},
    'plot_bgcolor': '#FAFAFA'
}))
figure.update_layout(width=800, height=500)
figure.update_layout(margin=dict(t=25))

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/donor_gift_new.png')

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/donor_gift_en_or.png')


## Basic transformations> Date, lower case, stop words removal

#Date

df_en['Date'] = pd.to_datetime(df_en['date_str'], format='%Y-%m-%d')

df_en.columns

#lower.case
textual_columns = ['recipient_org', 'recipient', 'donor', 'gift_type','delivery_type','Description', 'gift_owner']
df_en[textual_columns] = df_en[textual_columns].apply(lambda x: x.str.lower())


# Remove stopwords from the textual columns
stop_words = set(stopwords.words('english'))  # Set the language of stopwords
df_en[textual_columns] = df_en[textual_columns].apply(lambda x: x.apply(lambda y: ' '.join([word for word in str(y).split() if word not in stop_words])))






###############Recipients and organizations and owners / 3 graphs
df = df_en

# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['gift_owner'])]['gift_owner']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:4]))

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift donor'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}, 'showgrid': True,'gridcolor': 'lightgray'},
    'plot_bgcolor': 'white'
}))
figure.update_layout(width=800, height=500)
figure.update_layout(margin=dict(t=25))

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/owner_gift_en_new.png')


pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/owner_gift_en.png')
