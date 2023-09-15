import spacy
import pandas as pd

df = pd.read_csv('/Users/jelena/Desktop/Python Projects/gifts/data/df_final.csv')
df.columns
# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Function to extract objects/entities from a text using spaCy
def extract_objects(text):
    doc = nlp(text)
    objects = [entity.text for entity in doc.ents if entity.label_ == 'PRODUCT']
    return objects


def extract_object_names(text):
    if pd.notnull(text):  # Check for null values
        doc = nlp(text)
        objects = []
        for ent in doc.ents:
            if ent.label_ == "GPE":  # GPE represents geopolitical entity (e.g., countries)
                objects.append(ent.text)
        return objects
    else:
        return ''

# Apply the function to the 'Description' column and store the results in 'objects' column
df['object'] = df['Description'].apply(extract_objects)

df[['object','Description']]

df[df['objects'] != '[]']



def get_subject_phrase(doc):
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def get_object_phrase(doc):
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

list = ['']
for sentence in df['Description']:
    doc = nlp(sentence)
    subject_phrase = get_subject_phrase(doc)
    object_phrase = get_object_phrase(doc)
    print(subject_phrase)
    print(object_phrase)