import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy
from spacy.lang.sl import Slovenian
import seaborn as sns
import seaborn.objects as so
import dtale
import statsmodels
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


data = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/df_final.csv')

data.columns
#who receives and who gets the gift



###############Recipients and organizations and owners / 3 graphs: recipient, gift_owner, recipient_org
df = data

# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['recipient'])]['recipient']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:2]))

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Gift recipient'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}, 'showgrid': True,'gridcolor': 'lightgray'},
    'plot_bgcolor': 'white'
}))
figure.update_layout(width=800, height=500)
figure.update_layout(margin=dict(t=25))
pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/g_rec_en_new.png')


pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_rec_en.png')


#########
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
    'xaxis': {'title': {'text': 'Gift owner'}, 'titlefont': {'size': 18}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}, 'showgrid': True,'gridcolor': 'lightgray'},
    'plot_bgcolor': 'white'
}))
figure.update_layout(width=800, height=500)
figure.update_layout(margin=dict(t=25))


pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_owner_en.png')


#########
# Assuming you have a DataFrame 'df' and a column named 'Naziv'
s = df[~pd.isnull(df['recipient_org'])]['recipient_org']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:15]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:2]))

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Organization received the gift'}, 'titlefont': {'size': 18}, 'tickfont': {'size': 14}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 18}, 'showgrid': True,'gridcolor': 'lightgray', 'tickfont': {'size': 14}},
    'plot_bgcolor': 'white'
}))
figure.update_layout(width=800, height=400)
figure.update_layout(margin=dict(t=25))

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/g_org_en_new.png')

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_org_en.png')

###### combine plots

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd

# Assuming you have a DataFrame 'df' and a column named 'recipient'
s_recipient = df[~pd.isnull(df['recipient'])]['recipient']
chart_recipient = pd.value_counts(s_recipient).to_frame(name='data')
chart_recipient.index.name = 'labels'
chart_recipient = chart_recipient.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart_recipient = chart_recipient[:15]
chart_recipient['labels'] = chart_recipient['labels'].apply(lambda x: ' '.join(x.split()[:2]))
charts_recipient = [go.Bar(x=chart_recipient['labels'].values, y=chart_recipient['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure_recipient = go.Figure(data=charts_recipient, layout=go.Layout({
    'barmode': 'group',
    #'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},
    'xaxis': {'title': {'text': 'Gift recipient'}, 'titlefont': {'size': 16}, 'tickfont': {'size': 16}},
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 16}, 'showgrid': True,'gridcolor': 'lightgray', 'tickfont': {'size': 16}},
    'plot_bgcolor': 'white'
}))
figure_recipient.update_layout(width=400, height=400)
figure_recipient.update_layout(margin=dict(t=25))

# Assuming you have a DataFrame 'df' and a column named 'recipient_org'
s_org = df[~pd.isnull(df['recipient_org'])]['recipient_org']
chart_org = pd.value_counts(s_org).to_frame(name='data')
chart_org.index.name = 'labels'
chart_org = chart_org.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart_org = chart_org[:15]
chart_org['labels'] = chart_org['labels'].apply(lambda x: ' '.join(x.split()[:2]))
charts_org = [go.Bar(x=chart_org['labels'].values, y=chart_org['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure_org = go.Figure(data=charts_org, layout=go.Layout({
    'barmode': 'group',
    #'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},
    'xaxis': {'title': {'text': 'Organization received the gift'}, 'titlefont': {'size': 16}, 'tickfont': {'size': 16}},
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 16}, 'showgrid': True,'gridcolor': 'lightgray', 'tickfont': {'size': 16}},
    'plot_bgcolor': 'white'
}))
figure_org.update_layout(width=400, height=400)
figure_org.update_layout(margin=dict(t=25))

# Create subplots with 1 row and 2 columns
fig = make_subplots(rows=1, cols=2)


# Add the second plot to the first column
fig.add_trace(figure_org['data'][0], row=1, col=1)
fig.update_xaxes(title_text=' ', title_font=dict(size=16), row=1, col=2)
fig.update_yaxes(title_text=' ', title_font=dict(size=16), showgrid=True, gridcolor='lightgray', row=1, col=2)
fig.update_traces(showlegend=False)  # Remove legend for the second subplot

# Add the first plot to the second column
fig.add_trace(figure_recipient['data'][0], row=1, col=2)
fig.update_xaxes(title_text=' ', title_font=dict(size=16), row=1, col=1)
fig.update_yaxes(title_text=' ', title_font=dict(size=16), showgrid=True, gridcolor='lightgray', row=1, col=1)
fig.update_traces(showlegend=False)  # Remove legend for the second subplot


# Add annotation for (A) and (B)
fig.add_annotation(text="(A) Gift recipient (organization)", xref="paper", yref="paper", x=0.02, y=1.15, showarrow=False, font=dict(size=16))
fig.add_annotation(text="(B) Gift recipient", xref="paper", yref="paper", x=0.6, y=1.15, showarrow=False, font=dict(size=16))

# Update layout for the entire figure
fig.update_layout(
    #legend=dict(orientation='h'),
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor='white',
    width=1000, height=400,
)

# Save the figure as an image
pio.write_image(fig, '/Users/jelena/Desktop/Python Projects/gifts/plots/gift_rec_combined_plots.png')
pio.write_image(fig, '/Users/jelena/Desktop/Python Projects/gifts/plots/gift_rec_combined_plots.pdf')

# Show the figure
fig.show()



dtale.show(df)
############### donors

df = data


#############################################

df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

s = df[~pd.isnull(df['donor'])]['donor']
chart = pd.value_counts(s.str.split(expand=True).stack())
chart = chart.to_frame(name='data').sort_index()
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:23]
charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'xaxis': {'title': {'text': 'Gift donors (words used in the description)'}, 'titlefont': {'size': 16}},
    'yaxis': {'title': {'text': 'Frequency'}, 'titlefont': {'size': 16}, 'showgrid': True,'gridcolor': 'lightgray'},
    'plot_bgcolor': 'white'
}))
figure.update_layout(width=800, height=400)
figure.update_layout(margin=dict(t=25))

# Create the figure
pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/donors_words_new.png')

#pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/donors_words.png')
# Show the figure
figure.show()
