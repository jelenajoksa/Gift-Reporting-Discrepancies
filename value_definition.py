import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from spacy.lang.sl import Slovenian
import seaborn as sns
import seaborn.objects as so
import

dataset = pd.read_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/gifts2.csv')

type(dataset)



import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer





# Define a list of sentences to group
sentences = dataset['Način določitve vrednosti'].to_list()

nlp = Slovenian()

# Create a dictionary to keep track of unique stems and their corresponding original sentences
stem_dict = {}

# Loop through each sentence
for sentence in sentences:
    # Tokenize the sentence into words and remove punctuation and whitespace
    words = [token.text for token in nlp(sentence) if not token.is_punct and not token.is_space]
    # Stem each word in the sentence
    stemmed_words = [token.lemma_ for token in nlp(' '.join(words))]
    # Join the stemmed words back into a sentence
    stemmed_sentence = ' '.join(stemmed_words)
    # Check if the stemmed sentence is already in the dictionary
    if stemmed_sentence in stem_dict:
        # If it is, add the original sentence to the list of corresponding sentences
        stem_dict[stemmed_sentence].append(sentence)
    else:
        # If it's not, add the stemmed sentence as a new key and add the original sentence as the first corresponding sentence
        stem_dict[stemmed_sentence] = [sentence]
for i, (stemmed_sentence, sentence_list) in enumerate(stem_dict.items()):
    print(f"Group {i+1}: {sentence_list[0]}")
    for sentence in sentence_list:
        print('\t' + sentence)
# Create a list to store the group number for each sentence
group_numbers = []

# Loop through each sentence in the original list
for sentence in sentences:
    # Tokenize the sentence into words and remove punctuation and whitespace
    words = [token.text for token in nlp(sentence) if not token.is_punct and not token.is_space]
    # Stem each word in the sentence
    stemmed_words = [token.lemma_ for token in nlp(' '.join(words))]
    # Join the stemmed words back into a sentence
    stemmed_sentence = ' '.join(stemmed_words)
    # Find the group number for the stemmed sentence in the stem_dict
    group_number = [i+1 for i, (stemmed_sentence_, sentence_list) in enumerate(stem_dict.items()) if stemmed_sentence_ == stemmed_sentence][0]
    # Append the group number to the group_numbers list
    group_numbers.append(group_number)

# Create a pandas DataFrame with the original sentences and their corresponding group numbers
df = pd.DataFrame({'sentences': sentences, 'group_numbers': group_numbers})

# Print the DataFrame
print(df)

dataset['value_definition'] = df['group_numbers']

group_mapping = {1: 'Value found online',
                 2: 'Impossible to estimate as a nonprofessional',
                 3: 'Personal nonprofessional estimate',
                 4: 'General estimate',
                 5: 'Personal nonprofessional estimate',
                 6: 'Value defined on product',
                 7: 'Value defined by an expert',
                 8: 'Value defined on product',
                 9: 'Value found online',
                 10: 'General estimate',
                 11: 'Personal nonprofessional estimate',
                 12: 'Value of picture estimated comparing with other artists',
                 13: 'Value found online',
                 14: 'Value defined on product',
                 15: 'Impossible to estimate as a nonprofessional',
                 16: 'Value defined on product',
                 17: 'Impossible to estimate as a nonprofessional'
                 }


group_mapping2 = {1: 'online',
                 2: 'impossible',
                 3: 'personal',
                 4: 'general',
                 5: 'personal',
                 6: 'visible',
                 7: 'expert',
                 8: 'visible',
                 9: 'online',
                 10: 'general',
                 11: 'personal',
                 12: 'general',
                 13: 'online',
                 14: 'visible',
                 15: 'impossible',
                 16: 'visible',
                 17: 'impossible'
                 }
dataset['value_def_2'] = df['group_numbers'].replace(group_mapping2)

dataset['Vrednost']

sns.catplot(data=dataset, x="value_def_2", y="Vrednost")
plt.xticks(rotation=45)
plt.ylim(1000,30000)
plt.show()

import statistics

statistics.mode(dataset['Vrednost'])
statistics.median(dataset['Vrednost'])
statistics.quantiles(dataset['Vrednost'],n = 10)
(dataset[['Darovalec','Opis','value_def_2']][dataset['Vrednost']==10]).to_csv('/Users/jelena/Desktop/Python Projects/Learning/gifts/10eur.csv')
dataset[['Opis','value_def_2']][dataset['Vrednost']==20]

dataset.columns