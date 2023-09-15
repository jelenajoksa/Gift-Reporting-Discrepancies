import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.graph_objs as go
import plotly.io as pio
import warnings
import plotly.express as px


data = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/df_final.csv')
df_all = data
data['Year_Month'] = df_all['Year_Month']

data.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_final.csv', index=False, header=True)

data.columns
# Group the DataFrame by year and count the number of records
df_all['Date'] = pd.to_datetime(df_all['Date'])
max(df_all['Date'])
df_all[df_all['Date'] > '2023-06-01']
df_all['Date'] = df_all['Date'].replace(pd.Timestamp('2107-04-21 00:00:00'), pd.Timestamp('2017-04-21 00:00:00'))
df_all['Date'] = df_all['Date'].replace(pd.Timestamp('2023-11-13 00:00:00'), pd.Timestamp('2022-11-13 00:00:00'))

import ast
df_all['donor_geo'] = df_all['donor_geo'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df_all = df_all[df_all['donor_geo'].map(len) > 0]


df_filtered = df_all[df_all['donor_geo'].apply(lambda x: bool(x))]

df_grouped = df_filtered.groupby(df_filtered['Date'].dt.year).size().reset_index(name='Count')

# Get the top 3 countries for each year and create a 'Other' category
df_countries_top3 = df_all.explode('donor_geo')
df_countries_top3 = df_countries_top3.groupby(df_countries_top3['Date'].dt.year)['donor_geo'].value_counts().groupby(level=0).head(3).reset_index(name='Count')
df_countries_top3 = df_countries_top3.pivot(index='Date', columns='donor_geo', values='Count')

countries = df_countries_top3.columns.tolist()

# Determine the number of unique countries
num_countries = len(countries)

# Get the color palette
color_palette = px.colors.qualitative.Light24

# Create the stacked bar plot
fig = go.Figure()
for i, column in enumerate(df_countries_top3.columns):
    color_index = i % num_countries
    fig.add_trace(go.Bar(x=df_countries_top3.index, y=df_countries_top3[column], name=column, marker_color=color_palette[color_index]))

# Customize the layout
fig.update_layout(
    barmode='stack',
    xaxis={'title': 'Year'},
    yaxis={'title': 'Number of gifts'},
    legend={'orientation': 'v'},
    plot_bgcolor='white',
    showlegend=True
)

# Show the plot
fig.show()
pio.write_image(fig, '/Users/jelena/Desktop/Python Projects/gifts/plots/g_time_donor_new.png')


pio.write_image(fig, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_time_donor.png')



df_all.loc[:, 'Year'] = df_all['Date'].dt.year
df_all.loc[:, 'Month'] = df_all['Date'].dt.month

# Group the data by year and month and count the number of records
df_month = df_all.groupby(df_all['Month'])['Date'].count().reset_index(name='Count')

# Create a bar plot for monthly records
fig_month = px.bar(df_month, x='Month', y='Count', labels={'Month': 'Month', 'Count': 'Number of Records'}, title='Number of Records per Month')

# Group by year and calculate the count of records
df_yearly = df_all.groupby(df_all['Year'])['Date'].count().reset_index(name='Count')

# Create a bar plot for yearly records
fig_yearly = px.bar(df_yearly, x='Year', y='Count', labels={'Year': 'Year', 'Count': 'Number of Records'}, title='Number of Records per Year')

# Create a line plot for yearly records
fig_yearly.add_trace(px.line(df_yearly, x='Year', y='Count').data[0])

# Create a scatter plot for yearly records
fig_yearly.update_traces(mode='markers', marker=dict(color=df_yearly['Count'], colorscale='Viridis'))


fig_monthly.show()

fig_yearly.show()
pio.write_image(fig_yearly, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_time_y.png')


df_all.loc[:,'Year_Month'] = df_all['Date'].dt.strftime('%Y-%m')

df_monthly = df_all.groupby(df_all['Year_Month'])['Date'].count().reset_index(name='Count')

# Create a bar plot for monthly records
fig_monthly = px.bar(df_monthly, x='Year_Month', y='Count', labels={'Year_Month': 'Month', 'Count': 'Number of Records'})

line_trace = px.line(df_monthly, x='Year_Month', y='Count').data[0]
line_trace.line.color = '#DE542C'  # Set the line color to red

# Add the line trace to the figure
fig_monthly.add_trace(line_trace)

fig_monthly.update_layout(
    plot_bgcolor='white',  # Set the background color to white
    yaxis={'title': 'Number of gifts', 'showgrid': True, 'gridcolor': 'lightgray'}  # Show gridlines with a light gray color on the y-axis
)
fig_monthly.update_layout(
    xaxis={'tickmode': 'array', 'tickvals': ['2013-01', '2015-01', '2017-01', '2019-01', '2021-01', '2023-01']}
)

fig_monthly.update_layout(width=800, height=400)
fig_monthly.update_layout(margin=dict(t=25))
# Show the plots
fig_monthly.show()
pio.write_image(fig_monthly, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_time_m_all.png')


######## year ######



df_all.loc[:,'Year'] = df_all['Date'].dt.strftime('%Y')

df_y = df_all.groupby(df_all['Year'])['Date'].count().reset_index(name='Count')

# Create a bar plot for monthly records
fig_y = px.bar(df_y, x='Year', y='Count', labels={'Year': 'Year', 'Count': 'Number of Records'})
line_trace = px.line(df_y, x='Year', y='Count').data[0]
line_trace.line.color = '#DE542C'  # Set the line color to red

# Add the line trace to the figure
fig_y.add_trace(line_trace)

fig_y.update_layout(
    plot_bgcolor='white',  # Set the background color to white
    yaxis={'title': 'Number of gifts', 'showgrid': True, 'gridcolor': 'lightgray'}  # Show gridlines with a light gray color on the y-axis
)
fig_y.update_layout(
    xaxis={'tickmode': 'array', 'tickvals': ['2013', '2014','2015', '2016','2017', '2018','2019','2020', '2021','2022', '2023']}
)
fig_y.update_layout(width=800, height=400)
fig_y.update_layout(margin=dict(t=25))
# Show the plots
fig_y.show()
pio.write_image(fig_y, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_time_y_all.png')


df_all.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_final.csv', index=False, header=True)

(df_all['Year_Month']).unique()

#most popular month

df_month = df_all.groupby(df_all['Month'])['Date'].count().reset_index(name='Count')

fig_month = px.bar(df_month, x='Month', y='Count', labels={'Month': 'Month', 'Count': 'Number of Records'})


fig_month.update_layout(
    plot_bgcolor='white',  # Set the background color to white
    yaxis={'title': 'Number of gifts', 'showgrid': True, 'gridcolor': 'lightgray'}
)
fig_month.update_layout(
    xaxis={'tickmode': 'array', 'tickvals': df_month['Month'], 'ticktext': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']}
)

fig_month.update_layout(width=800, height=400)
fig_month.update_layout(margin=dict(t=25))

pio.write_image(fig_month, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_month.png')

