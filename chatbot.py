import numpy as np
import aiml
import sqlite3
import re
import pandas as pd
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class DatabaseHelper():
    '''A helper library for processing and accessing the SQL database in memory'''

    def __init__(self, file_name: str):
        # self._df = pd.read_sql_query('SELECT * FROM BoardGames', conn)
        self._df = pd.read_csv(file_name)
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
        result = self.proccessed_df.loc[self.proccessed_df['stats.average'] >= score]
        if len(result.index) > 0:
            return result.iloc[:, :-1]  # Do not return 'details.searchname'
        else:
            return None

    def search_by_similarity(self, query: str, min_confidence: float) -> pd.DataFrame:
        # Create a column of TF-IDF vectors on details.description
        description_column = [query] + \
            self.proccessed_df['details.description'].tolist()
        tfidfs = TfidfVectorizer().fit_transform(description_column)
        cosine_similarities = linear_kernel(tfidfs[0:1], tfidfs).flatten()
        # top 6 as one entry is query
        similar_indexes = cosine_similarities.argsort()[:-6:-1]
        accurate_indexes = [
            i-1 for i in similar_indexes if cosine_similarities[i] > min_confidence]  # i-1 to account for query padding the cosine indexes
        similar_entries = self.proccessed_df.iloc[accurate_indexes[1:]]
        return similar_entries, cosine_similarities[similar_indexes].tolist()[1:]

    @staticmethod
    def pretty_print_dataframe(dataframe: pd.DataFrame, columns=None, confidence_scores=None):
        if columns is None:
            columns = {
                'details.name': 'Name',
                'attributes.boardgamecategory': 'Categories',
                'stats.average': 'Rating'
            }
        data_to_print: pd.DataFrame = dataframe[columns]
        i = 0
        for index, entry in data_to_print.iterrows():
            output = ''
            for key, name in columns.items():
                output += f'{name}: {entry[key]}\n'
            if confidence_scores is not None:
                output += f'With a similarity of {round(confidence_scores[i]* 100, ndigits=2) }%\n'
            i = i+1
            print(f'{output}')


def print_top_entries(dataframe: pd.DataFrame, message: str, amount=5):
    '''
    Prints the top X entries by unweighted score, where X is defined by amount. 
    '''
    top = dataframe.sort_values(by='stats.average', ascending=False)[0:amount]
    print(message)
    print(f'The top {amount} are:')
    DatabaseHelper.pretty_print_dataframe(top)


class ResponseAgent(Enum):
    AIML = 'aiml'
    IMAGE = 'image'
    TOY = 'toy'
    STS = 'sts'
    RL = 'rl'


# Create database helper
db = DatabaseHelper('database.csv')

# Create AIML Agent
agent = aiml.Kernel()
agent.bootstrap(learnFiles='boardgames.xml')  # Add link to AIML file


# CLI chatbot
while True:
    try:
        question = input("-> ")
    except(KeyboardInterrupt, EOFError) as e:
        print("Goodbye")
        break

    # Response agent decision logic here
    response_agent = ResponseAgent.AIML

    if response_agent == ResponseAgent.AIML:
        response = agent.respond(question)
        if len(response) is 0:
            continue
    elif response_agent == ResponseAgent.IMAGE:
        raise NotImplementedError
    elif response_agent == ResponseAgent.TOY:
        raise NotImplementedError
    elif response_agent == ResponseAgent.STS:
        raise NotImplementedError
    elif response_agent == ResponseAgent.RL:
        raise NotImplementedError

    if response[0] == '^':  # ^CommandSubcommand$Param
        params = response.split('$')
        command_block = params[0]  # ^CommandSubcommand
        command = command_block[1:]  # CommandSubcommand
        if command[0] == 'e':  # End chat
            print(params[1])
            break
        elif command[0] == 's':  # Search DB
            sub_command = command[1]
            if sub_command == 'n':
                results = db.search_by_name(params[1])
                if results is not None:
                    DatabaseHelper.pretty_print_dataframe(results, columns={
                        'details.name': 'Name',
                        'attributes.boardgamecategory': 'Categories',
                        'details.description': 'Description',
                        'stats.average': 'Rating'})
                else:
                    print('Sorry, no results were found')
            elif sub_command == 'p':
                results = db.search_by_minplayers(int(params[1]))
                if results is not None:
                    message = f'{len(results)} games support {params[1]} or more players.'
                    print_top_entries(results, message)
                else:
                    print('Sorry, no results were found')
            elif sub_command == 'y':
                results = db.search_by_yearpublished(int(params[1]))
                if results is not None:
                    message = f'{len(results)} games have been published since {params[1]}.'
                    print_top_entries(results, message)
                else:
                    print('Sorry, no results were found')
            elif sub_command == 'c':
                items = re.split(' |,', params[1])
                results = db.search_by_specific_category(items)
                if results is not None:
                    message = f'{len(results)} games have the category {params[1]}.'
                    print_top_entries(results, message)
                else:
                    print('Sorry, no results were found')
            elif sub_command == 'g':
                items = re.split(' |,', params[1])
                results = db.search_by_multiple_category(items)
                if results is not None:
                    message = f'{len(results)} games have the categories {params[1]}.'
                    print_top_entries(results, message)
                else:
                    print('Sorry, no results were found')
            elif sub_command == 't':
                results = db.search_by_playtime(int(params[1]))
                if results is not None:
                    message = f'{len(results)} games have playtimes less than {params[1]} hours.'
                    print_top_entries(results, message)
                else:
                    print('Sorry, no results were found')
            elif sub_command == 's':
                results = db.search_by_score(float(params[1]))
                if results is not None:
                    message = f'{len(results)} games have scores of {params[1]} or more.'
                    print_top_entries(results, message)
                else:
                    print('Sorry, no results were found')
            else:
                print('Sorry, something went wrong.')
        elif command[0] == 'x':  # run cosine similarity across database
            similar, confidence_score = db.search_by_similarity(params[1], 0.2)
            print('The closest matches I have are:')
            DatabaseHelper.pretty_print_dataframe(
                similar, confidence_scores=confidence_score)
    else:
        print(response)
