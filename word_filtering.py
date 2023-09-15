df2['Book_Flag'] = df2['Description'].str.contains('book|books', case=False, regex=True)
df2['Painging_Flag'] = df2['Description'].str.contains('painting|picture|oil in canvas|graphic|oil canvas', case=False, regex=True)
df2['Bowl_Flag'] = df2['Description'].str.contains('bowl|bowls', case=False, regex=True)
df2['Carpet_Flag'] = df2['Description'].str.contains('carpet|carpets', case=False, regex=True)
df2['Plate_Flag'] = df2['Description'].str.contains('^(?!.*plated).*plate[s]?', case=False, regex=True)
df2['Set_Flag'] = df2['Description'].str.contains('set|cups|service|teapot|tea cup|samovar|cup', case=False, regex=True)
df2['Ticket_Flag'] = df2['Description'].str.contains('ticket|tickets', case=False, regex=True)
df2['Bottle_Flag'] = df2['Description'].str.contains('bottle|bottles', case=False, regex=True)
df2['Bottle_Flag'] = df2['Description'].str.contains('bottle|bottles|cognac|viliamovka|whiskey|wine', case=False, regex=True)
df2['Bottle_Flag'] = df2['Description'].str.contains('^(?!.*gift bag).*(bottle|bottles|cognac|viliamovka|whiskey|wine|gin|vodka|champagne|chivas)', case=False, regex=True)
#df2['Pen_Flag'] = df2['Description'].str.contains('^(?!.*gift bag).*(pen|pens)', case=False, regex=True)
df2['Pen_Flag'] = df2['Description'].str.contains('^(?!.*gift bag|.*book|.*books|.*opener).*pen|pens', case=False, regex=True)
df2['Giftbag_Flag'] = df2['Description'].str.contains('gift bag|gift box|gift basket|new year|chocolate|praline|sweets|dates|candies|teas|calendar|wall clock|juice|cookies', case=False, regex=True)
df2['Coin_Flag'] = df2['Description'].str.contains('coin|coins|medallion|medallions', case=False, regex=True)
df2['Vase_Flag'] = df2['Description'].str.contains('vase|vases', case=False, regex=True)
df2['Luxury_Flag'] = df2['Description'].str.contains('scarf|', case=False, regex=True)

columns = df2.columns[26:-16]


df_objects = pd.DataFrame({'Column': columns})

df_objects['True_Count'] = df2[columns].sum().values
df_objects['Sum_Value'] = df2.loc[df2[columns].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns].any(axis=1), columns].idxmax(axis=1)).sum().reindex(columns, fill_value=0).values
df_objects['Mean_Value'] = df2.loc[df2[columns].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns].any(axis=1), columns].idxmax(axis=1)).apply(lambda x: x.mean() if len(x) > 0 else 0).reindex(columns, fill_value=0).values
df_objects['Median_Value'] = df2.loc[df2[columns].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns].any(axis=1), columns].idxmax(axis=1)).apply(lambda x: x.median() if len(x) > 0 else 0).reindex(columns, fill_value=0).values
df_objects['Max_Value'] = df2.loc[df2[columns].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns].any(axis=1), columns].idxmax(axis=1)).max().reindex(columns, fill_value=0).values
df_objects['Third_Quartile'] = df2.loc[df2[columns].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns].any(axis=1), columns].idxmax(axis=1)).quantile(0.75).reindex(columns, fill_value=0).values

sorted_df = df_objects.sort_values('True_Count', ascending=False)

import matplotlib.pyplot as plt

# Select the columns for boxplot
boxplot_columns = sorted_df['Column']

boxplot_labels = [col.replace('_Flag', '') for col in boxplot_columns]

fig, ax = plt.subplots(figsize=(10, 5))

# Iterate over the columns and create boxplots
for i, col in enumerate(boxplot_columns):
    bp = ax.boxplot(df2.loc[df2[col], 'Vrednost'], widths = 0.6, positions=[i],  patch_artist=True)

    # Customize boxplot colors
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['fliers'][0].set(marker='o', color='gray', alpha=0.5, linewidth=0.5)  # Adjust outlier properties
    bp['caps'][0].set(color='black', linewidth=0.5)
    bp['caps'][1].set(color='black', linewidth=0.5)
    bp['whiskers'][0].set(color='black', linewidth=0.5)
    bp['whiskers'][1].set(color='black', linewidth=0.5)

    #ax.boxplot(df2.loc[df2[col], 'Vrednost'], positions=[i])
    ax.set_xticks(range(len(boxplot_columns)))
    ax.set_xticklabels(boxplot_labels, rotation=90, fontsize=14)
    ax.set_ylabel('Gift values', fontsize=14)
    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1000, 10000, 100000])  # Set custom ticks
    ax.set_yticklabels([ '10 EUR','100 EUR', '1K EUR', '10K EUR','100K EUR'],
                       fontsize=12)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=0.8)
    #ax.axhline(y=131, color='red', linestyle='--', linewidth=0.5)
    ax.axhline(y=1000, color='lightblue', linestyle='--', linewidth=0.5)
    ax.axhline(y=10000, color='lightblue', linestyle='--', linewidth=0.5)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
fig.savefig('/Users/jelena/Desktop/Python Projects/gifts/plots/object_values.png')

for i, col in enumerate(boxplot_columns):
    ax.boxplot(df2.loc[df2[col], 'Vrednost'], positions=[i])
    ax.set_xticks(range(len(boxplot_columns)))
    ax.set_xticklabels(boxplot_labels, rotation=90, fontsize=14)
    ax.set_ylabel('Gift values', fontsize = 14)
    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1000, 10000])  # Set custom ticks
    ax.set_yticklabels(['10 EUR', '100 EUR', '1000 EUR', '10 000 EUR'],
                       fontsize=14)

# Adjust spacing between subplots
plt.tight_layout()

df_objects.sort_values('True_Count', ascending=False)

columns2 = df2.columns[56:-1]

df_materials = pd.DataFrame({'Column': columns2})


df2 = df2.drop(df2[df2['Column'] == 'Model_Flag'].index)
df2 = df2.drop(df2[df2['Column'] == 'Replica_Flag'].index)
df2 = df2.drop(df2[df2['Column'] == 'Sculpture_Flag'].index)


df2 = df2.drop('marbel_Flag', axis=1)
df2 = df2.drop('peal_Flag', axis=1)
df2 = df2.drop('Model_Flag', axis=1)
df2 = df2.drop('Replica_Flag', axis=1)
df2 = df2.drop('Sculpture_Flag', axis=1)
df2 = df2.drop('Medallion_Flag', axis=1)
df2 = df2.drop('Medallion_Flag', axis=1)

df_materials.sort_values('True_Count', ascending=False)


df_materials = pd.DataFrame({'Column': columns2})

df_materials['True_Count'] = df2[columns2].sum().values
df_materials['Sum_Value'] = df2.loc[df2[columns2].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns2].any(axis=1), columns2].idxmax(axis=1)).sum().reindex(columns2, fill_value=0).values
df_materials['Mean_Value'] = df2.loc[df2[columns2].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns2].any(axis=1), columns2].idxmax(axis=1)).apply(lambda x: x.mean() if len(x) > 0 else 0).reindex(columns2, fill_value=0).values
df_materials['Median_Value'] = df2.loc[df2[columns2].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns2].any(axis=1), columns2].idxmax(axis=1)).apply(lambda x: x.median() if len(x) > 0 else 0).reindex(columns2, fill_value=0).values
df_materials['Max_Value'] = df2.loc[df2[columns2].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns2].any(axis=1), columns2].idxmax(axis=1)).max().reindex(columns2, fill_value=0).values
df_materials['Third_Quartile'] = df2.loc[df2[columns2].any(axis=1), 'Vrednost'].groupby(df2.loc[df2[columns2].any(axis=1), columns2].idxmax(axis=1)).quantile(0.75).reindex(columns2, fill_value=0).values

sorted_df_mat = df_materials.sort_values('True_Count', ascending=False)




import matplotlib.pyplot as plt

# Select the columns for boxplot
boxplot_columns = sorted_df_mat['Column']

boxplot_labels = [col.replace('_Flag', '') for col in boxplot_columns]

fig, ax = plt.subplots(figsize=(10, 5))

# Iterate over the columns and create boxplots
for i, col in enumerate(boxplot_columns):
    bp = ax.boxplot(df2.loc[df2[col], 'Vrednost'], widths = 0.6, positions=[i],  patch_artist=True)

    # Customize boxplot colors
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['fliers'][0].set(marker='o', color='gray', alpha=0.5, linewidth=0.5)  # Adjust outlier properties
    bp['caps'][0].set(color='black', linewidth=0.5)
    bp['caps'][1].set(color='black', linewidth=0.5)
    bp['whiskers'][0].set(color='black', linewidth=0.5)
    bp['whiskers'][1].set(color='black', linewidth=0.5)

    #ax.boxplot(df2.loc[df2[col], 'Vrednost'], positions=[i])
    ax.set_xticks(range(len(boxplot_columns)))
    ax.set_xticklabels(boxplot_labels, rotation=90, fontsize=14)
    ax.set_ylabel('Gift values', fontsize=14)
    ax.set_yscale('log')
    ax.set_yticks([10, 100, 1000, 10000, 100000])  # Set custom ticks
    ax.set_yticklabels([ '10 EUR','100 EUR', '1K EUR', '10K EUR','100K EUR'],
                       fontsize=12)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=0.8)
    #ax.axhline(y=131, color='red', linestyle='--', linewidth=0.5)
    ax.axhline(y=1000, color='lightblue', linestyle='--', linewidth=0.5)
    ax.axhline(y=10000, color='lightblue', linestyle='--', linewidth=0.5)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
fig.savefig('/Users/jelena/Desktop/Python Projects/gifts/plots/object_mat_values.png')

not_clustered_df = df2[(df2['Vase_Flag'] == 0) & (df2['Coin_Flag'] == 0) & (df2['Giftbag_Flag'] == 0) & (df2['Book_Flag'] == 0)
    & (df2['Pen_Flag'] == 0) & (df2['Bottle_Flag'] == 0) & (df2['Ticket_Flag'] == 0) & (df2['Set_Flag'] == 0)
    & (df2['Plate_Flag'] == 0) & (df2['Carpet_Flag'] == 0) & (df2['Painging_Flag'] == 0) & (df2['Bowl_Flag'] == 0)]

(not_clustered_df['Description']).sample(10)

df2.columns

len(df2[df2['Book_Flag']==True]['Description'])
(df2[df2['Book_Flag']==1]['Description']).sample(10)


#### Second round

df2['Cuff_Flag'] = df2['Description'].str.contains('cuff|cuffs|tie', case=False, regex=True)
df2['Medallion_Flag'] = df2['Description'].str.contains('medallion|medallions', case=False, regex=True)
df2['Plaque_Flag'] = df2['Description'].str.contains('plaque|plaques', case=False, regex=True)
df2['Monograph_Flag'] = df2['Description'].str.contains('monograph|monographs', case=False, regex=True)
df2['Saint_Flag'] = df2['Description'].str.contains('saint|saints', case=False, regex=True)
df2['Frame_Flag'] = df2['Description'].str.contains('^(?!.*framed).*frame[s]?', case=False, regex=True)
df2['Statue_Flag'] = df2['Description'].str.contains('statue|statues|model|models|sculpture|sculptures|figurine|replica|replicas', case=False, regex=True)
df2['Model_Flag'] = df2['Description'].str.contains('model|models', case=False, regex=True)
df2['Replica_Flag'] = df2['Description'].str.contains('replica|replicas', case=False, regex=True)
df2['Sculpture_Flag'] = df2['Description'].str.contains('sculpture|sculptures|figurine', case=False, regex=True)
df2['Watch_Flag'] = df2['Description'].str.contains('watch|wristwatch', case=False, regex=True)
df2['Photo_Flag'] = df2['Description'].str.contains('photo|image|photos|images', case=False, regex=True)
df2['CoatArm_Flag'] = df2['Description'].str.contains('coat|coats', case=False, regex=True)
df2['Award_Flag'] = df2['Description'].str.contains('award|awards|certificate', case=False, regex=True)
df2['Huawei_Flag'] = df2['Description'].str.contains('huawei|mobile|tablet|charger|air pad|headphones', case=False, regex=True)


not_clustered_df_2 = df2[(df2['Vase_Flag'] == 0) & (df2['Coin_Flag'] == 0) & (df2['Giftbag_Flag'] == 0) & (df2['Book_Flag'] == 0)
    & (df2['Pen_Flag'] == 0) & (df2['Bottle_Flag'] == 0) & (df2['Ticket_Flag'] == 0) & (df2['Set_Flag'] == 0)
    & (df2['Plate_Flag'] == 0) & (df2['Carpet_Flag'] == 0) & (df2['Painging_Flag'] == 0) & (df2['Bowl_Flag'] == 0) & (df2['Cuff_Flag'] == 0) & (df2['Medallion_Flag'] == 0)  & (df2['Monograph_Flag'] == 0)
    & (df2['Saint_Flag'] == 0) & (df2['Frame_Flag'] == 0) & (df2['Statue_Flag'] == 0) & (df2['Model_Flag'] == 0)
    & (df2['Replica_Flag'] == 0) & (df2['Plaque_Flag'] == 0) & (df2['Sculpture_Flag'] == 0) & (df2['Watch_Flag'] == 0)
    & (df2['Photo_Flag'] == 0) & (df2['CoatArm_Flag'] == 0) & (df2['Award_Flag'] == 0) & (df2['Huawei_Flag'] == 0)]



######## CLUSTERING not_clustered_df


opis_en = not_clustered_df['Description']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(opis_en)

feature_names = vectorizer.get_feature_names_out()

# Get the vocabulary (unique words)
#vocabulary = vectorizer.vocabulary_

# Get the number of unique words
#num_unique_words = len(feature_names)
#print("Number of unique words:", num_unique_words)

# Perform k-means clustering
num_clusters = 10 # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters, random_state=1)   #42
kmeans.fit(X)

# Get the cluster labels for each expression
cluster_labels = kmeans.labels_

not_clustered_df = not_clustered_df.drop('Cluster_Labels', axis=1)

# Add the cluster labels back to the dataframe
not_clustered_df['Cluster_Labels'] = cluster_labels

# Print the clusters and their corresponding expressions
#for cluster_id in range(num_clusters):
 #   cluster_expressions = df2[df2['Cluster_Labels'] == cluster_id]['Description'].tolist()
  #  print(f"Cluster {cluster_id}:")
  #  for expression in cluster_expressions:
  #      print(expression)
 #   print()


cluster_dict = {}

# Populate the cluster dictionary with the values from 'Opis_EN' for each cluster
unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_data = not_clustered_df.loc[cluster_indices, 'Description'].tolist()
    cluster_dict[cluster] = cluster_data

# Print the cluster dictionary
for cluster, data in cluster_dict.items():
    print(f"Cluster {cluster}:")
    print(data)
    print(len(data))
    print()

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

# Evaluate the clustering performance
silhouette_avg = silhouette_score(X, not_clustered_df['Cluster_Labels'])
calinski_harabasz_score = calinski_harabasz_score(X.toarray(), not_clustered_df['Cluster_Labels'])

print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score}")

silhouette_scores = silhouette_samples(X, cluster_labels)


unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_silhouette_score = silhouette_scores[cluster_indices].mean()
    print(f"Cluster {cluster}: Silhouette Score = {cluster_silhouette_score}")

###### common expression of the cluster


from collections import Counter

# Assuming you have performed clustering and obtained cluster labels

# Create an empty dictionary to store the most common two-word expression for each cluster
common_expression_dict = {}
len(cluster_indices)
# Iterate over each cluster
unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_silhouette_score = silhouette_scores[cluster_indices].mean()
    cluster_data = not_clustered_df.loc[cluster_indices, 'Description']

    # Concatenate all the two-word expressions in the cluster
    two_word_expressions = ' '.join(cluster_data.apply(lambda x: ' '.join(x.split()[:3])))

    # Find the most common two-word expression using Counter
    common_expression = Counter(two_word_expressions.split()).most_common(1)[0][0]

    # Store the most common two-word expression in the dictionary
    common_expression_dict[cluster] = common_expression
    print(common_expression)
    print(cluster_silhouette_score)
    print(len(cluster_data))

not_clustered_zero = not_clustered_df[not_clustered_df['Cluster_Labels']==0]


from collections import Counter

# Concatenate all descriptions into a single string
all_descriptions = ' '.join(not_clustered_df_2['Description'].values)

# Split the string into individual words
words = all_descriptions.split()

# Count the frequency of each word
word_counts = Counter(words)

# Get the five most common words
most_common_words = word_counts.most_common(40)

# Extract only the words (without their counts)
most_common_words = [word for word, count in most_common_words]

print(most_common_words)

not_clustered_df_2[not_clustered_df_2['Description'].str.contains('gdf2['Book_Flag'] = df2['Description'].str.contains('book|books', case=False, regex=True)
df2['Painging_Flag'] = df2['Description'].str.contains('painting|picture|oil in canvas|graphic|oil canvas', case=False, regex=True)
df2['Bowl_Flag'] = df2['Description'].str.contains('bowl|bowls', case=False, regex=True)
df2['Carpet_Flag'] = df2['Description'].str.contains('carpet|carpets', case=False, regex=True)
df2['Plate_Flag'] = df2['Description'].str.contains('^(?!.*plated).*plate[s]?', case=False, regex=True)
df2['Set_Flag'] = df2['Description'].str.contains('set|cups|service|teapot|tea cup|samovar|cup', case=False, regex=True)
df2['Ticket_Flag'] = df2['Description'].str.contains('ticket|tickets', case=False, regex=True)
df2['Bottle_Flag'] = df2['Description'].str.contains('bottle|bottles', case=False, regex=True)
df2['Bottle_Flag'] = df2['Description'].str.contains('bottle|bottles|cognac|viliamovka|whiskey|wine', case=False, regex=True)
df2['Bottle_Flag'] = df2['Description'].str.contains('^(?!.*gift bag).*(bottle|bottles|cognac|viliamovka|whiskey|wine|gin|vodka|champagne|chivas)', case=False, regex=True)
df2['Pen_Flag'] = df2['Description'].str.contains('^(?!.*(gift bag)|.*book|.*books|.*opener).*pen|pens', case=False, regex=True)
df2['Giftbag_Flag'] = df2['Description'].str.contains('gift bag|gift box|gift basket|new year|chocolate|praline|sweets|dates|candies|teas|calendar|wall clock|juice|cookies', case=False, regex=True)
df2['Coin_Flag'] = df2['Description'].str.contains('coin|coins', case=False, regex=True)
df2['Vase_Flag'] = df2['Description'].str.contains('vase|vases', case=False, regex=True)
df2['Luxury_Flag'] = df2['Description'].str.contains('^(?!.*book).*scarf|jewelry|brooch|wallet|pearls|pearl|necklace|bracelet|ring|earrings|glasses?', case=False, regex=True)
df2['Wooden_Flag'] = df2['Description'].str.contains('^(?!.*(plate)|.*carpet|.*bowl|.*vase|.*box|.*frame|.*statue|.*sculpture|.*plaque|.*monograph).*wooden', case=False, regex=True)



not_clustered_df = df2[(df2['Vase_Flag'] == 0) & (df2['Coin_Flag'] == 0) & (df2['Giftbag_Flag'] == 0) & (df2['Book_Flag'] == 0)
    & (df2['Pen_Flag'] == 0) & (df2['Bottle_Flag'] == 0) & (df2['Ticket_Flag'] == 0) & (df2['Set_Flag'] == 0)
    & (df2['Plate_Flag'] == 0) & (df2['Carpet_Flag'] == 0) & (df2['Painging_Flag'] == 0) & (df2['Bowl_Flag'] == 0)]

(not_clustered_df['Description']).sample(10)

len(df2[df2['Luxury_Flag']==True]['Description'])
(df2[df2['Luxury_Flag']==1]['Description']).sample(10)


#### Second round

df2['Cuff_Flag'] = df2['Description'].str.contains('cuff|cuffs|tie', case=False, regex=True)
df2['Medallion_Flag'] = df2['Description'].str.contains('medallion|medallions', case=False, regex=True)
df2['Plaque_Flag'] = df2['Description'].str.contains('plaque|plaques', case=False, regex=True)
df2['Monograph_Flag'] = df2['Description'].str.contains('monograph|monographs', case=False, regex=True)
df2['Saint_Flag'] = df2['Description'].str.contains('saint|saints', case=False, regex=True)
df2['Frame_Flag'] = df2['Description'].str.contains('^(?!.*framed).*frame[s]?', case=False, regex=True)
df2['Statue_Flag'] = df2['Description'].str.contains('statue', case=False, regex=True)
df2['Model_Flag'] = df2['Description'].str.contains('model|models', case=False, regex=True)
df2['Replica_Flag'] = df2['Description'].str.contains('replica|replicas', case=False, regex=True)
df2['Sculpture_Flag'] = df2['Description'].str.contains('sculpture|sculptures', case=False, regex=True)
df2['Watch_Flag'] = df2['Description'].str.contains('watch|wristwatch', case=False, regex=True)
df2['Photo_Flag'] = df2['Description'].str.contains('photo|image|photos|images', case=False, regex=True)
df2['CoatArm_Flag'] = df2['Description'].str.contains('coat|coats', case=False, regex=True)
df2['Award_Flag'] = df2['Description'].str.contains('award|awards|certificate', case=False, regex=True)
df2['Huawei_Flag'] = df2['Description'].str.contains('huawei|mobile|tablet|charger|air pad|headphones', case=False, regex=True)
df2['Box_Flag'] = df2['Description'].str.contains('^(?!.*(gift box)|.*(book box)|.*(tea box)).*box', case=False, regex=True)


not_clustered_df_2 = df2[(df2['Vase_Flag'] == 0) & (df2['Coin_Flag'] == 0) & (df2['Giftbag_Flag'] == 0) & (df2['Book_Flag'] == 0)
    & (df2['Pen_Flag'] == 0) & (df2['Bottle_Flag'] == 0) & (df2['Ticket_Flag'] == 0) & (df2['Set_Flag'] == 0)
    & (df2['Plate_Flag'] == 0) & (df2['Carpet_Flag'] == 0) & (df2['Painging_Flag'] == 0) & (df2['Bowl_Flag'] == 0) & (df2['Cuff_Flag'] == 0) & (df2['Medallion_Flag'] == 0)  & (df2['Monograph_Flag'] == 0)
    & (df2['Saint_Flag'] == 0) & (df2['Frame_Flag'] == 0) & (df2['Statue_Flag'] == 0) & (df2['Model_Flag'] == 0)
    & (df2['Replica_Flag'] == 0) & (df2['Plaque_Flag'] == 0) & (df2['Sculpture_Flag'] == 0) & (df2['Watch_Flag'] == 0)
    & (df2['Photo_Flag'] == 0) & (df2['CoatArm_Flag'] == 0) & (df2['Award_Flag'] == 0) & (df2['Huawei_Flag'] == 0)   & (df2['Luxury_Flag'] == 0) & (df2['Box_Flag'] == 0) &  (df2['Wooden_Flag'] == 0)]



######## CLUSTERING not_clustered_df


opis_en = not_clustered_df['Description']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(opis_en)

feature_names = vectorizer.get_feature_names_out()

# Get the vocabulary (unique words)
#vocabulary = vectorizer.vocabulary_

# Get the number of unique words
#num_unique_words = len(feature_names)
#print("Number of unique words:", num_unique_words)

# Perform k-means clustering
num_clusters = 10 # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters, random_state=1)   #42
kmeans.fit(X)

# Get the cluster labels for each expression
cluster_labels = kmeans.labels_

not_clustered_df = not_clustered_df.drop('Cluster_Labels', axis=1)

# Add the cluster labels back to the dataframe
not_clustered_df['Cluster_Labels'] = cluster_labels

# Print the clusters and their corresponding expressions
#for cluster_id in range(num_clusters):
 #   cluster_expressions = df2[df2['Cluster_Labels'] == cluster_id]['Description'].tolist()
  #  print(f"Cluster {cluster_id}:")
  #  for expression in cluster_expressions:
  #      print(expression)
 #   print()


cluster_dict = {}

# Populate the cluster dictionary with the values from 'Opis_EN' for each cluster
unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_data = not_clustered_df.loc[cluster_indices, 'Description'].tolist()
    cluster_dict[cluster] = cluster_data

# Print the cluster dictionary
for cluster, data in cluster_dict.items():
    print(f"Cluster {cluster}:")
    print(data)
    print(len(data))
    print()

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

# Evaluate the clustering performance
silhouette_avg = silhouette_score(X, not_clustered_df['Cluster_Labels'])
calinski_harabasz_score = calinski_harabasz_score(X.toarray(), not_clustered_df['Cluster_Labels'])

print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score}")

silhouette_scores = silhouette_samples(X, cluster_labels)


unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_silhouette_score = silhouette_scores[cluster_indices].mean()
    print(f"Cluster {cluster}: Silhouette Score = {cluster_silhouette_score}")

###### common expression of the cluster


from collections import Counter

# Assuming you have performed clustering and obtained cluster labels

# Create an empty dictionary to store the most common two-word expression for each cluster
common_expression_dict = {}
len(cluster_indices)
# Iterate over each cluster
unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_silhouette_score = silhouette_scores[cluster_indices].mean()
    cluster_data = not_clustered_df.loc[cluster_indices, 'Description']

    # Concatenate all the two-word expressions in the cluster
    two_word_expressions = ' '.join(cluster_data.apply(lambda x: ' '.join(x.split()[:3])))

    # Find the most common two-word expression using Counter
    common_expression = Counter(two_word_expressions.split()).most_common(1)[0][0]

    # Store the most common two-word expression in the dictionary
    common_expression_dict[cluster] = common_expression
    print(common_expression)
    print(cluster_silhouette_score)
    print(len(cluster_data))

not_clustered_zero = not_clustered_df[not_clustered_df['Cluster_Labels']==0]


from collections import Counter

# Concatenate all descriptions into a single string
all_descriptions = ' '.join(not_clustered_df_2['Description'].values)

# Split the string into individual words
words = all_descriptions.split()

# Count the frequency of each word
word_counts = Counter(words)

# Get the five most common words
most_common_words = word_counts.most_common(40)

# Extract only the words (without their counts)
most_common_words = [word for word, count in most_common_words]

print(most_common_words)
‹
len(not_clustered_df_2[not_clustered_df_2['Description'].str.contains('silver', case=False, regex=True)]['Description'])
df2[df2['Description'].str.contains('medallion', case=False, regex=True)]['Description']


df2['Luxury_Flag'] = df2['Description'].str.contains('scarf|jewelry|brooch|wallet|pearls|pearl|necklace|bracelet|ring|earrings', case=False, regex=True)


df2['Luxury_Flag'] = df2['Description'].str.contains('scarf|jewelry|brooch|wallet|pearls|pearl|', case=False, regex=True)
