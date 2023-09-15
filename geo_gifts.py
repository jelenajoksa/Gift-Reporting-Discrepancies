import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# Create a DataFrame with the country names and their counts
unique_countries = set(df['donor_geo'].explode().apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None).dropna())
print(all_countries)




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

country_codes = convert_to_country_code(all_countries)
print(country_codes)

type(country_codes)

country_counts = dict(Counter(country_codes))
print(country_counts)

data2 = {'Country Code': list(country_counts.keys()), 'Count': list(country_counts.values())}
df3 = pd.DataFrame(data2)

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my-app")  # Replace "my-app" with your own user agent
from geopy.geocoders import Nominatim

# Create a geocoder instance
from geopy.geocoders import Nominatim

geolocator = Nominatim()
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

# Print the updated DataFrame
print(df3)


import folium
from folium.plugins import MarkerCluster
#empty map


import folium
from folium.plugins import HeatMap

import folium
from folium.plugins import HeatMap

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
folium.TileLayer('cartodbdark_matter').add_to(world_map)

# Create a list of (latitude, longitude, weight) tuples for the HeatMap
heatmap_data = heatmap_df[['Latitude', 'Longitude', 'Count']].values.tolist()
max_count = heatmap_df['Count'].max()

# Create the HeatMap layer with customized options
HeatMap(heatmap_data,
         radius=15,  # Adjust the radius of each data point
         blur=10,  # Adjust the blur effect of the heatmap
         gradient={0.1: 'black', 0.3: 'orange',  1.0: 'red'},  # Adjust the color gradient
         min_opacity=0.2,
         max_opacity = 0.8 ).add_to(world_map)  # Adjust the maximum value for color normalization

# Show the map
world_map.save('plots/gift_heatmap.html')
world_map.save('plots/gift_heatmap.png')






