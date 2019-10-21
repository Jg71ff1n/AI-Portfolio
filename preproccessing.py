import pandas as pd


def dataframe_preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
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


def search_by_name(dataframe: pd.DataFrame, name: str) -> pd.DataFrame:
    '''
    Case-insensitive search of dataframe, based on the games name and the name provided.
    Returns an empty dataframe when no items are found.
    '''
    cleaned_search = name.lower()
    result = dataframe.loc[dataframe['details.searchname'] == cleaned_search]
    if len(result.index) > 0:
        return result.iloc[:, :-1]  # Do not return 'details.searchname'
    else:
        return None


def search_by_minplayers(dataframe: pd.DataFrame, players: int) -> pd.DataFrame:
    '''
    Searches the dataframe by minimum players, returns all entries with a minimum player amount >= amount provided.
    Returns an empty dataframe when no items are found.
    '''
    result = dataframe.loc[dataframe['details.minplayers'] >= players]
    if len(result.index) > 0:
        return result.iloc[:, :-1]  # Do not return 'details.searchname'
    else:
        return None


def search_by_yearpublished(dataframe: pd.DataFrame, year: int) -> pd.DataFrame:
    '''
    Searches the dataframe by publish date, returns all entries with a published date >= amount provided.
    Returns an empty dataframe when no items are found.
    '''
    result = dataframe.loc[dataframe['details.yearpublished'] >= year]
    if len(result.index) > 0:
        return result.iloc[:, :-1]  # Do not return 'details.searchname'
    else:
        return None


def search_by_specific_category(dataframe: pd.DataFrame, categories: list) -> pd.DataFrame:
    '''
    Searches the dataframe by a list of categories, returns all entries with a catergory matches all supplied list.
    Returns an empty dataframe when no items are found.
    '''
    expression = '(?=.*{})'
    result = dataframe.loc[dataframe['attributes.boardgamecategory'].str.contains(
        ''.join(expression.format(category) for category in categories), case=False)]
    if len(result.index) > 0:
        return result.iloc[:, :-1]  # Do not return 'details.searchname'
    else:
        return None


def search_by_multiple_category(dataframe: pd.DataFrame, categories: list) -> pd.DataFrame:
    '''
    Searches the dataframe by a list of categories, returns all entries with a catergory contained within supplied list.
    Returns an empty dataframe when no items are found.
    '''

    result = dataframe.loc[dataframe['attributes.boardgamecategory'].str.contains(
        '|'.join(categories), case=False)]
    if len(result.index) > 0:
        return result.iloc[:, :-1]  # Do not return 'details.searchname'
    else:
        return None


def search_by_playtime(dataframe: pd.DataFrame, playtime: float) -> pd.DataFrame:
    '''
    Searches the dataframe by max playtime, returns all entries with a maximum playtime <= amount provided.
    Returns an empty dataframe when no items are found.
    '''
    result = dataframe.loc[dataframe['details.maxplaytime'] <= playtime]
    if len(result.index) > 0:
        return result.iloc[:, :-1]  # Do not return 'details.searchname'
    else:
        return None


def search_by_score(dataframe: pd.DataFrame, score: float) -> pd.DataFrame:
    '''
    Searches the dataframe by weighted user scores, returns all entries with a bayes average >= score provided.
    Returns an empty dataframe when no items are found.
    '''
    result = dataframe.loc[dataframe['stats.bayesaverage'] >= score]
    if len(result.index) > 0:
        return result.iloc[:, :-1]  # Do not return 'details.searchname'
    else:
        return None
