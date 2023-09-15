df2['gold_Flag'] = df2['Description'].str.contains('gold|golden', case=False, regex=True)
df2['silver_Flag'] = df2['Description'].str.contains('silver|silvered', case=False, regex=True)
df2['jewelry_Flag'] = df2['Description'].str.contains('jewelry', case=False, regex=True)
df2['pearl_Flag'] = df2['Description'].str.contains('pearl|pearls', case=False, regex=True)
df2['wooden_Flag'] = df2['Description'].str.contains('wood|wooden', case=False, regex=True)
df2['glass_Flag'] = df2['Description'].str.contains('glass', case=False, regex=True)
df2['crystal_Flag'] = df2['Description'].str.contains('crystal', case=False, regex=True)
df2['porcelain_Flag'] = df2['Description'].str.contains('porcelain', case=False, regex=True)
df2['ceramic_Flag'] = df2['Description'].str.contains('ceramic', case=False, regex=True)
df2['stone_Flag'] = df2['Description'].str.contains('stone', case=False, regex=True)
df2['bronze_Flag'] = df2['Description'].str.contains('bronze', case=False, regex=True)
df2['leather_Flag'] = df2['Description'].str.contains('leather', case=False, regex=True)
df2['handmade_Flag'] = df2['Description'].str.contains('handmade|hand made', case=False, regex=True)
df2['marble_Flag'] = df2['Description'].str.contains('marble', case=False, regex=True)




len(df2[df2['handmade_Flag']==True]['Description'])
(df2[df2['gold_Flag']==1]['Description']).sample(10)





#### Brand extraction

import spacy

# Load the pre-trained English language model
nlp = spacy.load('en_core_web_sm')

# Function to extract brands from a text
def extract_brands(text):
    doc = nlp(text)
    brands = []
    for ent in doc.ents:
        if ent.label_ == 'PERSON':  # Filter entities labeled as organizations
            brands.append(ent.text)
    return brands

# Apply the function to the 'Description' column and create a new column 'Brands'
df2['Brands'] = df2['Description'].apply(extract_brands)

# Apply the function to the 'Description' column and create a new column 'Brands'
non_empty_brands = df2[df2['Brands'].astype(bool)]

non_empty_brands.sample(20)