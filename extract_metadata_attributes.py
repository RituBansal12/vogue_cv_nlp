import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from keybert import KeyBERT
import spacy

# Load NER pipeline (transformer-based)
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
# Load KeyBERT for keyphrase extraction
kw_model = KeyBERT()
# Load spaCy for noun phrase extraction
nlp = spacy.load("en_core_web_sm")

# Load your metadata
df = pd.read_csv("tabular_data/covers_metadata.csv")

def find_people(text):
    entities = ner(text)
    return list(set(ent['word'] for ent in entities if ent['entity_group'] == "PER"))

def find_brands(text):
    entities = ner(text)
    return list(set(ent['word'] for ent in entities if ent['entity_group'] == "ORG"))

def find_outfit_descriptions(text, top_n=10):
    # Use KeyBERT to extract keyphrases
    keyphrases = [kw[0] for kw in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)]
    # Use spaCy to filter for noun phrases (likely to be clothing items)
    doc = nlp(text)
    noun_phrases = set(chunk.text.lower() for chunk in doc.noun_chunks)
    # Return intersection of keyphrases and noun phrases, plus all keyphrases
    outfit_desc = set()
    for phrase in keyphrases:
        if phrase.lower() in noun_phrases or len(phrase.split()) == 1:
            outfit_desc.add(phrase)
    return list(outfit_desc)

rows = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = f"{row.get('Caption', '')} {row.get('Retail information', '')}"
    brands_found = find_brands(text)
    people_found = find_people(text)
    apparel_found = find_outfit_descriptions(text)
    rows.append({
        "date_column": row.get("date_column", ""),
        "brands": brands_found,
        "people": people_found,
        "outfit_descriptions": apparel_found
    })

out_df = pd.DataFrame(rows)
out_df.to_csv("tabular_data/covers_metadata_attributes.csv", index=False)
print("Saved NLP mentions to tabular_data/covers_metadata_attributes.csv")