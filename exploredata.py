import sqlite3
import pandas as pd

conn = sqlite3.connect('board-games-dataset/database.sqlite')
df = pd.read_sql_query('SELECT * FROM BoardGames', conn)

print(df.head())
print(df.shape)
conn.close()