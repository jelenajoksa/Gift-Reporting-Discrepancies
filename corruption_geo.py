corr_countries = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/countries_review.csv')

type(corr_countries)
corr_countries.columns
corr_countries.to_list()

# Assuming 'corr_countries' is your DataFrame
column_list = corr_countries['Countries'].tolist()


import pycountry
import re

def convert_to_country_code(countries):
    country_codes = []

    for country in countries:
        # Attempt to match the country name with pycountry's database
        try:
            country_obj = pycountry.countries.search_fuzzy(country)[0]
            country_code = country_obj.alpha_2
            country_codes.append(country_code)
        except LookupError:
            # If no match is found, skip the entry
            continue

    return country_codes

# Example usage

from collections import Counter

country_codes = convert_to_country_code(column_list)

country_counts = dict(Counter(country_codes))
print(country_counts)

data2 = {'Country Code': list(country_counts.keys()), 'Count': list(country_counts.values())}
df3 = pd.DataFrame(data2)

import numpy as np
def geolocate(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan

# Apply the function to your DataFrame and create latitude and longitude columns
df3['Latitude'] = df3['Country Code'].map(geolocate).apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
df3['Longitude'] = df3['Country Code'].map(geolocate).apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)








# Create an empty dataframe
import folium
from folium.plugins import HeatMap

# Create an empty dataframe
heatmap_df = pd.DataFrame(columns=['Latitude', 'Longitude', 'Count'])

df3 = df3.dropna(subset=['Latitude', 'Longitude'])

# Iterate over each row in the original dataframe
for _, row in df3.iterrows():
    count = row['Count']
    latitude = row['Latitude']
    longitude = row['Longitude']

    # Create duplicate rows based on the count
    duplicate_rows = pd.DataFrame({'Latitude': [latitude] * count,
                                   'Longitude': [longitude] * count,
                                   'Count': [1] * count})

    # Append the duplicate rows to the heatmap dataframe
    heatmap_df = heatmap_df.append(duplicate_rows, ignore_index=True)

# Create an empty map
world_map = folium.Map(tiles="cartodbpositron")
#folium.TileLayer('cartodbdark_matter').add_to(world_map)

# Create a list of (latitude, longitude, weight) tuples for the HeatMap
heatmap_data = heatmap_df[['Latitude', 'Longitude', 'Count']].values.tolist()
max_count = heatmap_df['Count'].max()

# Create the HeatMap layer with customized options
HeatMap(heatmap_data,
         radius=15,  # Adjust the radius of each data point
         blur=1,  # Adjust the blur effect of the heatmap
         gradient={0.3: 'orange', 0.6: 'blue',  0.8: 'red'},  # Adjust the color gradient
         min_opacity=0.2,
         max_opacity = 0.5 ).add_to(world_map)  # Adjust the maximum value for color normalization

# Show the map
world_map.save('plots/corr_heatmap.html')

