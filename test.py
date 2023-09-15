import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.graph_objs as go
import plotly.io as pio
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import dtale

data = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/gifts_en.csv')
df_all = data
df_all.columns
df_all['Opis_EN'].sample(5)


dtale.show(df_all)
