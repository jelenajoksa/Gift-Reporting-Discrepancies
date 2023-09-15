import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import plotly.graph_objs as go
import plotly.io as pio
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


data = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/df_final.csv')
df_all = data
df_all.columns
data['donor']

import spacy
import pandas as pd

# Load the pre-trained English language model in spaCy
nlp = spacy.load("en_core_web_sm")



###################




# Function to extract country names from a text using spaCy
def extract_country_names(text):
    if pd.notnull(text):  # Check for null values
        doc = nlp(text)
        countries = []
        for ent in doc.ents:
            if ent.label_ == "GPE":  # GPE represents geopolitical entity (e.g., countries)
                countries.append(ent.text)
        return countries
    else:
        return []

# Apply the function to the column in the DataFrame
df_all["donor_geo_2"] = df_all["donor"].apply(extract_country_names)

# Print the extracted country names
df2[['Countries_donors', 'donor_geo']]

df_all["Countries_donors2"] = df_all["donor_geo_2"].apply(lambda x: [c for c in x if c])

# Create a final list of unique countries
all_countries = [country for countries in df_all["Countries_donors2"] for country in countries]

# Print the final list of countries
print(all_countries)



country_counts = pd.Series(all_countries).value_counts().reset_index()
country_counts.columns = ['Country', 'Frequency']

top_countries = country_counts.head(25)


# Create a bar plot using Plotly
fig = go.Figure(data=go.Bar(x=top_countries['Country'], y=top_countries['Frequency'], marker={'color': '#DE542C'}))

fig.update_layout(
    xaxis=dict(title='Geopolitical entities (e.g. countries) of gift donors', titlefont=dict(size=18)),
    yaxis=dict(title='Frequency', titlefont=dict(size=18), showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white'
)
fig.update_layout(width=800, height=500, margin=dict(t=25))


fig.show()
pio.write_image(fig, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_country_donor_en.png')

# Show the plot
df_all.columns

df_all.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/df_final.csv', index=False, header=True)




#### NEW

data['Countries_donors']

data['Countries_donors_2'] = df_all["donor_geo_2"]

all_countries = [country for countries in data["Countries_donors"] for country in countries]
data.to_csv('/Users/jelena/Desktop/Python Projects/gifts/data/df_final.csv', index=False, header=True)



data["Countries_donors2"] = data["Countries_donors_2"].apply(lambda x: [c for c in x if c])

# Create a final list of unique countries
all_countries = [country for countries in data["Countries_donors2"] for country in countries]

# Print the final list of countries
print(all_countries)



country_counts = pd.Series(all_countries).value_counts().reset_index()
country_counts.columns = ['Country', 'Frequency']

top_countries = country_counts.head(25)


# Create a bar plot using Plotly
fig = go.Figure(data=go.Bar(x=top_countries['Country'], y=top_countries['Frequency'], marker={'color': '#DE542C'}))

fig.update_layout(
    xaxis=dict(title='Geopolitical entities (e.g. countries) of gift donors', titlefont=dict(size=18)),
    yaxis=dict(title='Frequency', titlefont=dict(size=18), showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white'
)
fig.update_layout(width=800, height=500, margin=dict(t=25))


fig.show()
pio.write_image(fig, '/Users/jelena/Desktop/Python Projects/Learning/gifts/plots/g_country_donor_en_new.png')
