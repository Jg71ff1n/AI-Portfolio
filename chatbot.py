import numpy as np
import aiml
import sqlite3
import pandas as pd
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DatabaseHelper():
    '''A helper library for processing and accessing the SQL database in memory'''

    def __init__(self, connection: sqlite3.Connection):
        self._df = pd.read_sql_query('SELECT * FROM BoardGames', conn)
        self.proccessed_df = self.dataframe_preprocess(self._df)

    def dataframe_preprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''Takes in the raw DataFrame from the SQL and processes it to clean up data and access only required columns and fields'''
        column_headings = ['game.id', 'game.type', 'details.description', 'details.image',
                           'details.maxplayers', 'details.maxplaytime', 'details.minage', 'details.minplayers', 'details.minplaytime',
                           'details.name', 'details.playingtime', 'details.thumbnail', 'details.yearpublished', 'attributes.boardgamecategory',
                           'stats.average', 'stats.bayesaverage']
        # Select useful columns
        selected_columns = dataframe[column_headings]
        # Remove expansions from dataset
        core_dataset = selected_columns[selected_columns['game.type']
                                        == 'boardgame']
        # NAN filling
        cleaned_data = core_dataset.apply(lambda x: x.fillna(
            0) if x.dtype.kind in 'biufc' else x.fillna('#'))
        # Lowercase names for ease of searching
        cleaned_data['details.searchname'] = cleaned_data['details.name'].str.lower()

        # Create a column of TF-IDF Scores on details.description
        cleaned_data['tf-idf'] = cleaned_data['details.description'].apply(
            lambda y: TfidfVectorizer(y))
        print(cleaned_data.head())
        return cleaned_data

    def search_by_name(self, name: str) -> pd.DataFrame:
        '''
        Case-insensitive search of dataframe, based on the games name and the name provided.
        Returns an empty dataframe when no items are found.
        '''
        cleaned_search = name.lower()
        result = self.proccessed_df.loc[self.proccessed_df['details.searchname']
                                        == cleaned_search]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_minplayers(self, players: int) -> pd.DataFrame:
        '''
        Searches the dataframe by minimum players, returns all entries with a minimum player amount >= amount provided.
        Returns an empty dataframe when no items are found.
        '''
        result = self.proccessed_df.loc[self.proccessed_df['details.minplayers'] >= players]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_yearpublished(self, year: int) -> pd.DataFrame:
        '''
        Searches the dataframe by publish date, returns all entries with a published date >= amount provided.
        Returns an empty dataframe when no items are found.
        '''
        result = self.proccessed_df.loc[self.proccessed_df['details.yearpublished'] >= year]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_specific_category(self, categories: list) -> pd.DataFrame:
        '''
        Searches the dataframe by a list of categories, returns all entries with a catergory matches all supplied list.
        Returns an empty dataframe when no items are found.
        '''
        expression = '(?=.*{})'
        result = self.proccessed_df.loc[self.proccessed_df['attributes.boardgamecategory'].str.contains(
            ''.join(expression.format(category) for category in categories), case=False)]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_multiple_category(self, categories: list) -> pd.DataFrame:
        '''
        Searches the dataframe by a list of categories, returns all entries with a catergory contained within supplied list.
        Returns an empty dataframe when no items are found.
        '''

        result = self.proccessed_df.loc[self.proccessed_df['attributes.boardgamecategory'].str.contains(
            '|'.join(categories), case=False)]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_playtime(self, playtime: float) -> pd.DataFrame:
        '''
        Searches the dataframe by max playtime, returns all entries with a maximum playtime <= amount provided.
        Returns an empty dataframe when no items are found.
        '''
        result = self.proccessed_df.loc[self.proccessed_df['details.maxplaytime'] <= playtime]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_score(self, score: float) -> pd.DataFrame:
        '''
        Searches the dataframe by weighted user scores, returns all entries with a bayes average >= score provided.
        Returns an empty dataframe when no items are found.
        '''
        result = self.proccessed_df.loc[self.proccessed_df['stats.bayesaverage'] >= score]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_similarity(self, score: float) -> pd.DataFrame:
        raise NotImplementedError()
        # search = self.proccessed_df.apply(lambda x: cosine_similarity(x['tf-idf'], score)
        

    @staticmethod
    def pretty_print_dataframe(dataframe: pd.DataFrame, columns=['details.name', 'details.description', 'attributes.boardgamecategory', 'stats.bayesaverage']):
        data_to_print: pd.DataFrame=dataframe[columns]
        for index, entry in data_to_print.iterrows():
            print(entry)


class ResponseAgent(Enum):
    AIML='aiml'
    IMAGE='image'
    TOY='toy'
    STS='sts'
    RL='rl'


# Import database
conn=sqlite3.connect('board-games-dataset/database.sqlite')
# Create database helper
db=DatabaseHelper(conn)

# Create AIML Agent
agent=aiml.Kernel()
agent.bootstrap(learnFiles='boardgames.xml')  # Add link to AIML file


# CLI chatbot
while True:
    try:
        question=input("-> ")
    except(KeyboardInterrupt, EOFError) as e:
        print("Goodbye")
        break

    # Response agent decision logic here
    response_agent=ResponseAgent.AIML

    if response_agent == ResponseAgent.AIML:
        response=agent.respond(question)
    elif response_agent == ResponseAgent.IMAGE:
        raise NotImplementedError
    elif response_agent == ResponseAgent.TOY:
        raise NotImplementedError
    elif response_agent == ResponseAgent.STS:
        raise NotImplementedError
    elif response_agent == ResponseAgent.RL:
        raise NotImplementedError

    if response[0] == '^':  # ^CommandSubcommand$Param
        params=response.split('$')
        command_block=params[0]  # ^CommandSubcommand
        command=command_block[1:]  # CommandSubcommand
        print(command)
        if command[0] == 'e':  # End chat
            print(params[1])
            break
        if command[0] == 's':  # Search DB
            sub_command=command[1]
            if sub_command == 'n':
                DatabaseHelper.pretty_print_dataframe(
                    db.search_by_name(params[1]))  # Print Details
            elif sub_command == 'p':
                db.search_by_minplayers(params[1])
            elif sub_command == 'y':
                db.search_by_yearpublished(int(params[1]))
            elif sub_command == 'c':
                raise NotImplementedError  # Need to segregate params into a list
                db.search_by_specific_category()
            elif sub_command == 'g':
                raise NotImplementedError  # Need to segregate params into a list
                db.search_by_multiple_category()
            elif sub_command == 't':
                # raise NotImplementedError  # Need to segregate params
                db.search_by_playtime(int(params[1]))
            elif sub_command == 's':
                db.search_by_score(float(params[1]))
                pass
            else:
                print('Sorry, something went wrong.')
        elif command[0] == 'X':
            # run similarity cosine across database
            print('Similarity run')
            input_tfidf=TfidfVectorizer(params[1])

    else:
        print(response)
