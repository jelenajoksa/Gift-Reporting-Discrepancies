import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the dataframe and select the 'Opis_EN' column
df2 = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/gifts_en.csv')

df_to_translate = df2.drop(['Opis', 'Opis_EN'], axis=1)
import pandas as pd
from nltk.stem import SnowballStemmer
import re
from deep_translator import GoogleTranslator

stemmer = SnowballStemmer("english")


def translate_text(text):
    if pd.isnull(text) or text == '':
        return ''

    translation = GoogleTranslator(source='sl', target='en').translate(text)
    return translation

for column in df_to_translate.columns:
    df_to_translate[column + '_EN'] = df_to_translate[column].astype(str).apply(translate_text)


df['Opis_EN'] = df['Opis'].astype(str).apply(translate_text)

df_to_translate.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_en.csv', index=False, header=True)

df_en = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_en.csv')
