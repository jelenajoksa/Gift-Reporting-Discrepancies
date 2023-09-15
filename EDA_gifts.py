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
from pandas_profiling import ProfileReport
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.io as pio

dataset = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/gifts2.csv')

len(dataset)
type(dataset)
dtale.show(dataset)

dataset.describe()
dataset.drop('Razlog za izročitev',axis=1)
import dtale

dtale.show(dataset)


type(dataset['Date'][0])

dataset['Date'] = pd.to_datetime(dataset['Datum'], format='%Y-%m-%d')


df = dataset

# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Naziv'])]['Naziv']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:10]

# Truncate x-axis labels after the second word
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Public officials'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, 'Name_top10.png')

################################################



# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Lastnik darila'])]['Lastnik darila']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:10]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift owner'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_owner.png')


################################################

df.columns


# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Razlog za izročitev'])]['Razlog za izročitev']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:10]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Occasion on which the gift is received'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_reason.png')





###############DAROVALEC
df = dataset

# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Darovalec'])]['Darovalec']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift donor'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/donor_gift.png')



#############################################

###############OPIS
df = dataset

# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Opis'])]['Opis']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:1]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift donor'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_desc.png')

########### WITH NLP ###################

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import spacy

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Assuming you have a DataFrame 'df' and a column named 'Opis'
s = df[~pd.isnull(df['Opis'])]['Opis']

# Convert words to lowercase
s = s.str.lower()

# Lemmatize words in the 'Opis' column
s = s.apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]

# Truncate x-axis labels after the second word
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift donor'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure
pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_desc_spacy.png')

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

df['Opis_EN'] = df['Opis'].astype(str).apply(translate_text)

#df.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/gifts_en.csv', index=False, header=True)

# Assuming you have a DataFrame 'df' and a column named 'Opis_EN'
s = df[~pd.isnull(df['Opis_EN'])]['Opis_EN']

# Rest of the code for creating the bar plot with variations of the same word as the same category
# ... (remaining code)


# Assuming you have a DataFrame 'df' and a column named 'Opis'
#s = df[~pd.isnull(df['Opis'])]['Opis']
from fuzzywuzzy import fuzz


# Apply stemming to reduce words to their base form
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

from rapidfuzz import fuzz, process

# Assuming you have a DataFrame 'df' and a column named 'Opis_EN'

from rapidfuzz import fuzz, process

def group_similar_strings(strings):
    grouped_strings = {}
    for string in strings:
        matched = False
        for key, value in grouped_strings.items():
            if fuzz.ratio(string, key) > 35:  # Adjust the similarity threshold as needed
                grouped_strings[key].append(string)
                matched = True
                break
        if not matched:
            grouped_strings[string] = [string]
    return grouped_strings

s = df[~pd.isnull(df['Opis_EN'])]['Opis_EN']

s = s.str.lower()
s = s.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))
#s = s.apply(lambda x: ' '.join(x.split()[:3]) )
grouped_strings = group_similar_strings(s)

# Extract the most important keyword from each group
grouped_keywords = {}
for key, value in grouped_strings.items():
    keyword = process.extractOne(key, value)[0]
    grouped_keywords[key] = keyword

# Count the frequency of grouped keywords
grouped_counts = {key: len(value) for key, value in grouped_keywords.items()}

chart = pd.DataFrame({'labels': list(grouped_counts.keys()), 'data': list(grouped_counts.values())})
chart = chart.sort_values(['data', 'labels'], ascending=[False, True])
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:4]))
chart = chart[:15]

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift Description (after transformation)'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_desc_3.png')


chart.iloc[0:10,:]


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

s = df[~pd.isnull(df['Opis_EN'])]['Opis_EN']
s = s.str.lower()
s = s.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))
s = s.apply(lambda x: ' '.join(x.split()[:3]) )

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) )

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift donor'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_desc_3.png')


#############################################

df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

s = df[~pd.isnull(df['Način določitve vrednosti'])]['Način določitve vrednosti']
chart = pd.value_counts(s.str.split(expand=True).stack())
chart = chart.to_frame(name='data').sort_index()
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:10]
charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'xaxis': {'title': {'text': 'How the price is estimated (words used in the description)'}, 'titlefont': {'size': 18}},
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/value_determination.png')

df = dataset


# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Način določitve vrednosti'])]['Način določitve vrednosti']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:10]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'How the price is estimated (whole description)'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_value_desc.png')


####### Vrednost  ##################

if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
	df = df.to_frame(index=False)

# remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

# main statistics
stats = df['Vrednost'].describe().to_frame().T
# sum
stats['sum'] = df['Vrednost'].sum()
# median
stats['median'] = df['Vrednost'].median()
# mode
mode = df['Vrednost'].mode().values
stats['mode'] = np.nan if len(mode) > 1 else mode[0]
# var
stats['var'] = df['Vrednost'].var()
# sem
stats['sem'] = df['Vrednost'].sem()
uniq_vals = df['Vrednost'].value_counts().sort_values(ascending=False)
uniq_vals.index.name = 'value'
uniq_vals.name = 'count'
uniq_vals = uniq_vals.reset_index()
uniq_vals.loc[:, 'type'] = 'int64'
sequential_diffs = df['Vrednost'].diff()
diff = diff[diff == diff]
min_diff = sequential_diffs.min()
max_diff = sequential_diffs.max()
avg_diff = sequential_diffs.mean()
diff_vals = sequential_diffs.value_counts().sort_values(ascending=False)

figure.update_layout(margin=dict(t=25))

# Create the figure

import plotly.express as px

# Create the box plot
figure = px.box(df, y='Vrednost', color="Lastnik darila")

# Customize the appearance of the plot
figure.update_layout(
    xaxis_title='',
    yaxis_title='Gift Values'
)

# Display the plot
figure.show()

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/value_boxplot.png')





q1 = df['Vrednost'].quantile(0.25)
q3 = df['Vrednost'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
filtered_df = df[(df['Vrednost'] >= lower_bound) & (df['Vrednost'] <= upper_bound)]

figure = go.Figure()
figure.add_trace(go.Box(
    y=filtered_df['Vrednost'],
    name=' ',
    boxpoints='all',
    jitter=0.3,
    whiskerwidth=0.2,
    marker=dict(size=4),
    line=dict(width=1)
))

figure.update_layout(
    xaxis_title='',
    yaxis_title='Gift Values (outliers removed)',
    showlegend=False
)

figure.show()

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/value_boxplot_no_out.png')

###################HISTOGRAM ON LOG SCALE##################


figure = go.Figure()
figure.add_trace(go.Histogram(
    x=df['Vrednost'],
    nbinsx=30,
    histnorm='probability',
    name='Histogram',
    marker=dict(color='steelblue'),
    opacity=0.75,
))

figure.update_layout(
    title='Histogram (Log Scale)',
    xaxis_title='Values',
    yaxis_title='Frequency',
    yaxis_type='log',  # Set the y-axis to log scale
    showlegend=False
)

figure.show()
pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/hist_gifts_log.png')


##############################################
df = dataset


# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Način izročitve'])]['Način izročitve']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:10]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'The way the gift is delivered (directly vs indirectly)'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/delivery_way.png')


df[(df['Vrednost']<=50)]

df[(df['Vrednost']>50)]

df[(df['Vrednost']>50) & (df['Vrednost']<=100)]

df[(df['Vrednost']>100) & (df['Vrednost']<=200)]

df[(df['Vrednost']>200) & (df['Vrednost']<=1000)]

df[(df['Vrednost']>1000)]


df[(df['Vrednost']>=1000) & (df['Vrednost']<10000)]

df[(df['Vrednost']>=10000) ]


# DISCLAIMER: 'df' refers to the data you passed in when calling 'dtale.show'

df = dataset


# Assuming you have a DataFrame 'df' and a column named 'Naziv'
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
    df = df.to_frame(index=False)

df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]

s = df[~pd.isnull(df['Vrednost'])]['Vrednost']
value_counts = s.value_counts().sort_values(ascending=False)[:25]

chart = pd.DataFrame({'labels': value_counts.index, 'data': value_counts.values})
chart = chart.sort_values('data', ascending=False)  # Sort by frequency in descending order
chart['labels'] = chart['labels'].astype(str)  # Convert labels to string

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]

figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'xaxis': {'title': {'text': 'Reported gift value'}, 'categoryorder': 'total descending'},
    'yaxis': {'title': {'text': 'Frequency'}}
}))
figure.update_layout(margin=dict(t=25))

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/value_gift_count.png')

#####################################


df = dataset


# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['Tip darila'])]['Tip darila']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:10]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift type (Protocol vs Occasional)'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_type.png')


################ GIFT DESC #############

df = df2


# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['cluster_name'])]['cluster_name']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:20]

# Truncate x-axis labels after the second word
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:2]))

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift Description (After Clustering)'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}}
}))

figure.update_layout(margin=dict(t=25))

# Create the figure

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/gift_desc.png')

