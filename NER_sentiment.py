import spacy
from textblob import TextBlob

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


def get_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity + 1