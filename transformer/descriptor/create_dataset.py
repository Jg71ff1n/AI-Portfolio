import pandas as pd
import re


def preprocess_sentence(sentence):
    sentence = str(sentence)
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


path = '/home/joe/GitDrive/AI-Portfolio/database.csv'

df = pd.read_csv(path)
df = df.dropna(subset=['details.name',  'details.description', 'attributes.boardgamecategory'])
titles_categories = pd.DataFrame()
titles_categories['titleCategories'] = df['details.name'] + \
    ' - ' + df['attributes.boardgamecategory']
titles_categories['description'] = df['details.description']
titles_categories = titles_categories.applymap(preprocess_sentence)
print(titles_categories.head())
titles_categories.to_csv('transformer-dataset.csv')
