import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms
import nltk
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataframe and select the 'Opis_EN' column
df2 = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/df_final.csv')

opis_en = df2['Description']
df2.columns

df2['occasion']
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
num_clusters = 100 # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters, random_state=1)   #42
kmeans.fit(X)

# Get the cluster labels for each expression
cluster_labels = kmeans.labels_

# Add the cluster labels back to the dataframe
df2['Cluster_Labels'] = cluster_labels

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
    cluster_data = df2.loc[cluster_indices, 'Description'].tolist()
    cluster_dict[cluster] = cluster_data

# Print the cluster dictionary
for cluster, data in cluster_dict.items():
    print(f"Cluster {cluster}:")
    print(data)
    print(len(data))
    print()

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

# Evaluate the clustering performance
silhouette_avg = silhouette_score(X, df2['Cluster_Labels'])
calinski_harabasz_score = calinski_harabasz_score(X.toarray(), df2['Cluster_Labels'])

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
    cluster_data = df2.loc[cluster_indices, 'Description']

    # Concatenate all the two-word expressions in the cluster
    two_word_expressions = ' '.join(cluster_data.apply(lambda x: ' '.join(x.split()[:3])))

    # Find the most common two-word expression using Counter
    common_expression = Counter(two_word_expressions.split()).most_common(1)[0][0]

    # Store the most common two-word expression in the dictionary
    common_expression_dict[cluster] = common_expression
    print(common_expression)
    print(cluster_silhouette_score)
    print(len(cluster_data))

# Print the most common two-word expression for each cluster
for cluster, expression in common_expression_dict.items():
    print(f"Cluster {cluster}: {expression}")
    print(f"Cluster {cluster}: Silhouette Score = {cluster_silhouette_score}")



######### Remove 'book' items   ############

from collections import defaultdict
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk import ngrams


#repeated_words = ['painting', 'art', 'picture', 'vase','pen', 'bowl', 'carpet', 'carpets', 'glass', 'plate', 'plates',  'ticket', 'tickets']

#luxury_words = ['horse', 'statue','gold', 'porcelain', 'watch', 'mobile', 'sculpture', 'pearl', 'silver', 'bronze', 'necklace', 'replica' , 'earrings',
#                'royal', 'lipizzaner', 'huawei', 'golden', 'tablet', 'marble','cash', 'money', 'handmade','special']


repeated_words = ['book', 'books']

repeated_words_found = []

# Iterate over each expression
repeated_words_counts = defaultdict(int)

# Iterate over each expression
for expression in opis_en:
    # Convert the expression to lowercase and split it into individual words
    words = expression.lower().split()

    # Check if any luxury word is present in the expression
    for repeated_word in repeated_words:
        if repeated_word in words:
            # Increment the count for the luxury word
            repeated_words_counts[repeated_word] += 1

# Print the counts of luxury words
for repeated_word, count in repeated_words_counts.items():
    print(f"{repeated_word}: {count}")



import pandas as pd

# Check if 'Description' column contains 'book' or 'books'
df2['Book_Flag'] = df2['Description'].str.contains('book|books', case=False, regex=True)

# Convert True/False values to 1/0
df2['Book_Flag'] = df2['Book_Flag'].astype(int)

df2[df2['Description'].str.contains('wooden', case=False, regex=True)]

df2[df2['Book_Flag'] ==1]

df3 = df2[df2['Book_Flag'] ==0]

df3['Description'] = df3['Description'].str.replace('-', '')

opis_en = df_all['Opis_EN']


df2['occasion']
# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(opis_en)

#feature_names = vectorizer.get_feature_names_out()

# Get the vocabulary (unique words)
#vocabulary = vectorizer.vocabulary_

# Get the number of unique words
#num_unique_words = len(feature_names)
#print("Number of unique words:", num_unique_words)

# Perform k-means clustering
num_clusters = 20 # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters, random_state=42)   #42
kmeans.fit(X)

# Get the cluster labels for each expression
cluster_labels = kmeans.labels_

df_all = df_all.drop('Cluster_Labels', axis=1)

# Add the cluster labels back to the dataframe
df_all['Cluster_Labels'] = cluster_labels

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
    cluster_data = df_all.loc[cluster_indices, 'Opis_EN'].tolist()
    cluster_dict[cluster] = cluster_data

# Print the cluster dictionary
for cluster, data in cluster_dict.items():
    print(f"Cluster {cluster}:")
    print(data)
    print(len(data))
    print()

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

# Evaluate the clustering performance
silhouette_avg = silhouette_score(X, df_all['Cluster_Labels'])
calinski_harabasz_score = calinski_harabasz_score(X.toarray(), df_all['Cluster_Labels'])

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
    cluster_data = df_all.loc[cluster_indices, 'Opis_EN']

    # Concatenate all the two-word expressions in the cluster
    two_word_expressions = ' '.join(cluster_data.apply(lambda x: ' '.join(x.split()[:3])))

    # Find the most common two-word expression using Counter
    common_expression = Counter(two_word_expressions.split()).most_common(1)[0][0]

    # Store the most common two-word expression in the dictionary
    common_expression_dict[cluster] = common_expression
    print(common_expression)
    print(len(cluster_data))
