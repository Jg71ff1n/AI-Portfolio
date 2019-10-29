import sqlite3
import pandas as pd
import preproccessing

conn = sqlite3.connect('board-games-dataset/database.sqlite')
df = pd.read_sql_query('SELECT * FROM BoardGames', conn)
proccessed_df = preproccessing.dataframe_preprocess(df)

# for col in proccessed_df.columns:
#     print("*", col, "-", "datatype:",
#           proccessed_df[col].dtype, ", unique values:", len(proccessed_df[col].unique()))

# print(proccessed_df.isnull().sum())
# name_search = search_by_name(proccessed_df, 'Monopoly')
# if name_search.empty:
#     print(f'No items found for key: lso')
# else:
#     print(name_search)

# print(proccessed_df.head())
# print(preproccessing.search_by_specific_category(proccessed_df, ['Adventure', 'Fantasy', 'Dice']))
conn.close()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.datasets import fetch_20newsgroups

twenty = fetch_20newsgroups()

tfidf = TfidfVectorizer().fit_transform(twenty.data)
first_row = tfidf[0:1]

cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
related_docs_indices = cosine_similarities.argsort()[:-5:-1]
print(f'Related indicies: {related_docs_indices}')
print(cosine_similarities[related_docs_indices])

for i in related_docs_indices:
    print(twenty.data[i])