import pandas as pd
import sqlite3


conn = sqlite3.connect('board-games-dataset/database.sqlite')
data_frame = pd.read_sql_query('SELECT * FROM BoardGames', conn)

column_headings = ['game.id', 'game.type', 'details.description', 'details.image',
                           'details.maxplayers', 'details.maxplaytime', 'details.minage', 'details.minplayers', 'details.minplaytime',
                           'details.name', 'details.playingtime', 'details.thumbnail', 'details.yearpublished', 'attributes.boardgamecategory',
                           'stats.average', 'stats.bayesaverage']
# Select useful columns
selected_columns = data_frame[column_headings]
# Remove expansions from dataset
core_dataset: pd.DataFrame = selected_columns[selected_columns['game.type'] == 'boardgame']

core_dataset = core_dataset[::4]

core_dataset.to_csv('export.csv')
