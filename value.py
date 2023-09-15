import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.graph_objs as go
import plotly.io as pio
import warnings
import plotly.express as px
import dtale


data = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_final.csv')

df = data

dtale.show(df)


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

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency', marker={'color': '#DE542C'})]

figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'xaxis': {'title': {'text': 'Reported gift value'}, 'categoryorder': 'total descending'},
    'yaxis': {'title': {'text': 'Frequency'}}
}))

figure.update_layout(
    plot_bgcolor='white',  # Set the background color to white
    yaxis={'title': 'Number of gifts', 'showgrid': True, 'gridcolor': 'lightgray'}
)
figure.update_layout(margin=dict(t=25))
figure.update_layout(width=800, height=400)

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/value_gift_.png')




#boxplot per owner

df.columns
import plotly.express as px

# Create the box plot
figure = px.box(df, y='Vrednost', color="recipient")

# Customize the appearance of the plot
figure.update_layout(
    xaxis_title='',
    yaxis_title='Gift values',
    margin={'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins

)

figure.update_layout(width=800, height=300)
figure.update_layout(margin=dict(t=25))
# Display the plot
figure.show()
pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/value_boxplot_rec.png')



figure = px.box(df, y='Vrednost', color="gift_owner")

# Customize the appearance of the plot
figure.update_layout(
    xaxis_title='',
    yaxis_title='Gift values',
    margin = {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins

)

figure.update_layout(width=800, height=300)
figure.update_layout(margin=dict(t=25))
# Display the plot
figure.show()
pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/value_boxplot.png')




###### value determination method

df = data
df.columns

#########

s = df[~pd.isnull(df['value_determination_method'])]['value_determination_method']

chart = pd.value_counts(s).to_frame(name='data')
chart.index.name = 'labels'
chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])
chart = chart[:5]

# Truncate x-axis labels after the second word
#chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:3]) + ' ...')
chart['labels'] = chart['labels'].apply(lambda x: ' '.join(x.split()[:4]))

charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency', marker={'color': '#DE542C'})]
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h'},
    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10},  # Adjust margins
    'xaxis': {'title': {'text': 'Value determination method'}, 'titlefont': {'size': 16}},  # Add ... after second word
    'yaxis': {'title': {'text': 'Number of gifts'}, 'titlefont': {'size': 16}, 'showgrid': True,'gridcolor': 'lightgray'},
    'plot_bgcolor': 'white'
}))
figure.update_layout(width=500, height=400)
figure.update_layout(margin=dict(t=25))

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/g_vdm_new.pdf')

pio.write_image(figure, '/Users/jelena/Desktop/Python Projects/gifts/plots/g_vdm_new.png')


