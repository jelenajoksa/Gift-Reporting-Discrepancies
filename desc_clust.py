import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms
import nltk
from nltk.corpus import stopwords


# Load the dataframe and select the 'Opis_EN' column
df2 = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/gifts_en.csv')
opis_en = df2['Opis_EN']

df2.columns
# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(opis_en)


feature_names = vectorizer.get_feature_names_out()

# Get the vocabulary (unique words)
vocabulary = vectorizer.vocabulary_

# Get the number of unique words
num_unique_words = len(feature_names)
print("Number of unique words:", num_unique_words)

# Perform k-means clustering
num_clusters = 20 # Number of clusters to create
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster labels for each expression
cluster_labels = kmeans.labels_

# Add the cluster labels back to the dataframe
df2['Cluster_Labels'] = cluster_labels

# Print the clusters and their corresponding expressions
for cluster_id in range(num_clusters):
    cluster_expressions = df2[df2['Cluster_Labels'] == cluster_id]['Opis_EN'].tolist()
    print(f"Cluster {cluster_id}:")
    for expression in cluster_expressions:
        print(expression)
    print()


cluster_dict = {}

# Populate the cluster dictionary with the values from 'Opis_EN' for each cluster
unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_data = df2.loc[cluster_indices, 'Opis_EN'].tolist()
    cluster_dict[cluster] = cluster_data

# Print the cluster dictionary
for cluster, data in cluster_dict.items():
    print(f"Cluster {cluster}:")
    print(data)
    print(len(data))
    print()

#CLUSTER NAMES

conditions = [
    df2['Cluster_Labels'] == 0,
    df2['Cluster_Labels'] == 1,
    df2['Cluster_Labels'] == 2,
    df2['Cluster_Labels'] == 3,
    df2['Cluster_Labels'] == 4,
    df2['Cluster_Labels'] == 5,
    df2['Cluster_Labels'] == 6,
    df2['Cluster_Labels'] == 7,
    df2['Cluster_Labels'] == 8,
    df2['Cluster_Labels'] == 9,
    df2['Cluster_Labels'] == 10,
    df2['Cluster_Labels'] == 11,
    df2['Cluster_Labels'] == 12,
    df2['Cluster_Labels'] == 13,
    df2['Cluster_Labels'] == 14,
    df2['Cluster_Labels'] == 15,
    df2['Cluster_Labels'] == 16,
    df2['Cluster_Labels'] == 17,
    df2['Cluster_Labels'] == 18,
    df2['Cluster_Labels'] == 19

    # Add more conditions and values as needed
]
values = [
    'LUXURY',
    'LUX_OTHER',
    'GLASS ITEMS (mostly luxury)',
    'BOWLS (of various materials and designs)',
    'BOOKS',
    'LUX_OTHER',
    'CARPETS (of various materials and designs)',
    'PLATES (porcelain, national motifs, crystal,...)',
    'SERVICE SETS (tea, coffee, handmade, porcelain,...)',
    'TICKETS (events, shows, theatre, cultural,...)',
    'PAINTINGS and PICTURES (oil on canvas, wood, graphics...)',
    'PAINTINGS and PICTURES (oil on canvas, wood, graphics...)',
    'BOTTLES (wine, whiskey, gin,... mostly come with more items)',
    'BOOKS',
    'PENS (fountain, mon blanc, engraved, Parker Sonnet, usually comes with more items)',
    'GIFT BAGS (mostly New Years Gifts)',
    'LUX_OTHER',
    'COINS (gold, collectors coin, memorial coins, silver, royal, national,...)',
    'GIFT BAGS (mostly New Years Gifts)',
    'VASES (moser, porcelain, ceramic,...)']

values2 = [0, 1,2,3,4,1,5,6,7,8,9,9,10,4,11,12,1,13,12,14]

    # Add more values corresponding to the conditions

# Create the new column based on the conditions and values
df2['cluster_name'] = np.select(conditions, values, default='Other')
df2['cluster_id'] = np.select(conditions, values2, default=0)


# Print the clusters and their corresponding expressions
for cluster_id in range(num_clusters):
    cluster_expressions = df2[df2['cluster_id'] == cluster_id]['Opis_EN'].tolist()
    print(f"Cluster {cluster_id}:")
    for expression in cluster_expressions:
        print(expression)
    print()


cluster_dict = {}

# Populate the cluster dictionary with the values from 'Opis_EN' for each cluster
unique_clusters = set(df2['cluster_id'] )
for cluster in unique_clusters:
    cluster_indices = df2['cluster_id']  == cluster
    cluster_data = df2.loc[cluster_indices, 'Opis_EN'].tolist()
    cluster_dict[cluster] = cluster_data

# Print the cluster dictionary
for cluster, data in cluster_dict.items():
    print(f"Cluster {cluster}:")
    print(data)
    print(len(data))
    print()


from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Evaluate the clustering performance
silhouette_avg = silhouette_score(X, df2['cluster_id'])
calinski_harabasz_score = calinski_harabasz_score(X.toarray(), df2['cluster_id'])

print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score}")


#EVALUATION OF EACH CLUSTER

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

unique_clusters = set(cluster_labels)
for cluster in unique_clusters:
    cluster_indices = cluster_labels == cluster
    cluster_silhouette_score = silhouette_scores[cluster_indices].mean()
    print(f"Cluster {cluster}: Silhouette Score = {cluster_silhouette_score}")



########


df2.to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/data/gifts_clusters.csv', index=False, header=True)


df2['cluster_name']
df2.columns

from collections import Counter
import re
from nltk.corpus import stopwords
from nltk import ngrams

filtered_values = df2[df2['cluster_id'] == 0]['Opis_EN'].tolist()
filtered_values2 = df2[df2['cluster_id'] == 1]['Opis_EN'].tolist()

# Print the filtered values
#print(filtered_values)
# Your list of expressions
expressions = filtered_values2

preprocessed_expressions = [re.sub(r'[^\w\s]', '', expression.lower()) for expression in expressions]

# Preprocess expressions: remove common characters and convert to lowercase
words = [word for expression in preprocessed_expressions for word in expression.split()]

# Remove common words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Count the occurrences of each word
word_counts = Counter(filtered_words)

# Get the top 5 most common words
top_words = word_counts.most_common(10)

# Extract the words from the top word-count tuples
top_common_words = [word for word, count in top_words]


least_common_words = word_counts.most_common()[:-20:-1]

# Extract the words from the least common word-count tuples
top_least_common_words = [word for word, count in least_common_words]

# Print the top 5 least common words
print(top_least_common_words)

# Print the top 5 most common words
print(top_common_words)

#SEARCH for LUXURY WORDS
from collections import defaultdict

expressions = filtered_values2

#expressions = filtered_values2

repeated_words = ['painting', 'art', 'picture', 'vase','pen', 'bowl', 'carpet', 'carpets', 'glass', 'plate', 'plates',  'ticket', 'tickets']

luxury_words = ['horse', 'statue','gold', 'porcelain', 'watch', 'mobile', 'sculpture', 'pearl', 'silver', 'bronze', 'necklace', 'replica' , 'earrings',
                'royal', 'lipizzaner', 'huawei', 'golden', 'tablet', 'marble','cash', 'money', 'handmade','special']

# Create an empty list to store the luxury words found in the expressions
luxury_words_found = []

# Iterate over each expression
luxury_word_counts = defaultdict(int)

# Iterate over each expression
for expression in expressions:
    # Convert the expression to lowercase and split it into individual words
    words = expression.lower().split()

    # Check if any luxury word is present in the expression
    for luxury_word in luxury_words:
        if luxury_word in words:
            # Increment the count for the luxury word
            luxury_word_counts[luxury_word] += 1

# Print the counts of luxury words
for luxury_word, count in luxury_word_counts.items():
    print(f"{luxury_word}: {count}")





############################# PRINT REPEATED WORDS


expressions = filtered_values2

#expressions = filtered_values2

repeated_words = ['painting', 'book', 'books', 'art', 'picture', 'vase', 'pen', 'bowl', 'carpet', 'carpets', 'glass', 'plate', 'plates',  'ticket', 'tickets']

# Create an empty list to store the luxury words found in the expressions
repeated_words_found = []

# Iterate over each expression
repeated_word_counts = defaultdict(int)

# Iterate over each expression
for expression in expressions:
    # Convert the expression to lowercase and split it into individual words
    words = expression.lower().split()

    # Check if any luxury word is present in the expression
    for repeated_word in repeated_words:
        if repeated_word in words:
            # Increment the count for the luxury word
            repeated_word_counts[repeated_word] += 1

# Print the counts of luxury words
for repeated_word, count in repeated_word_counts.items():
    print(f"{repeated_word}: {count}")
