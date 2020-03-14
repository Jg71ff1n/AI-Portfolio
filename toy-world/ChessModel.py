import nltk
import itertools
read_expression = nltk.sem.Expression.fromstring


class ChessModel():

    # Items of the model
    pieces = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
    grammar_file = 'board_games.fcfg'

    def __init__(self):
        # Keep a running list of assumptions
        self.assumptions = []
        # Add north is the opposite of south etc...
        self.assumptions.append(read_expression('north(x,y) <-> south(y,x)'))
        self.assumptions.append(read_expression('west(x,y) <-> east(y,x)'))
        self.assumptions.append(read_expression(
            'northeast(x,y) <-> southwest(y,x)'))
        self.assumptions.append(read_expression(
            'northwest(x,y) <-> southeast(y,x)'))
        # Locational differences are transative
        self.assumptions.append(read_expression(
            'north(x,y) & north(y,z) -> north(x,z)'))
        self.assumptions.append(read_expression(
            'west(x,y) & west(y,z) -> west(x,z)'))
        # Some directions are compositional
        self.assumptions.append(read_expression(
            'northeast(x,y) <-> north(x,y) & east(x,y)'))
        self.assumptions.append(read_expression(
            'northwest(x,y) <-> north(x,y) & west(x,y)'))
        self.assumptions.append(read_expression(
            'southeast(x,y) <-> south(x,y) & east(x,y)'))
        self.assumptions.append(read_expression(
            'southwest(x,y) <-> south(x,y) & west(x,y)'))
        # Need to ensure all pieces are treated as unique constants
        internal_pieces = itertools.combinations(self.pieces, 2)
        for piece in internal_pieces:
            self.assumptions.append(
                read_expression(f'{piece[0]} != {piece[1]}'))
        self.prover = nltk.Prover9()

    @staticmethod
    def get_satisfier(valuation: nltk.Valuation, satisfiers: set) -> list:
        '''
        Searches for the names of satisfing constants of set satisfiers in the provided valuation
        '''
        results = []
        for satisfier in satisfiers:
            for k, v in valuation.items():
                if len(v) == 1:
                    if v == satisfier:
                        results.append(k)
        return results

    def add_assumption(self, expression):
        '''
        Adds an expression to the list of assumptions of them model
        '''
        if type(expression) == str:
            expression = read_expression(expression)
        self.assumptions.append(expression)

    def process_input(self, user_input: str) -> str:
        '''
        Takes user input, parses it against the grammar file, selects the method to use, returns output of the action
        '''
        parser = nltk.load_parser(self.grammar_file, trace=0)
        tokens = user_input.lower().split()
        try:
            parsed_tokens = parser.parse(tokens)
        except ValueError as ve:
            return 'Sorry I do not understand that.'
        for tree in parsed_tokens:
            semantic = tree.label()['SEM']
        if type(semantic) == nltk.sem.logic.ExistsExpression:  # Looking at an exists relation
            return 'Yes' if self.prove_expression(semantic) else 'No'
        else:
            if 'x3' in str(semantic):  # Finding satisfiers for the query
                satisfier_strings = ', '.join(
                    str(x) for x in self.query_model(semantic)
                )
                return f'The answer to your question is: {satisfier_strings}'
            else:
                self.add_assumption(semantic)  # Add an assumption to the model
                return 'I have addded that knowledge to my model'

    def build_model(self):
        '''
        Builds the Mace model based on the current assumptions
        '''
        self.mace_model = nltk.MaceCommand(None, self.assumptions)
        self.mace_model.build_model()
        self.valuation = self.mace_model.valuation

    def prove_expression(self, expression) -> bool:
        '''
        Attempts to prove an expression based on the current assumptions
        '''
        return self.prover.prove(expression, self.assumptions)

    def query_model(self, expression) -> list:
        '''
        Queries the model based on the expression, returns any satisfing answers
        '''
        self.build_model()
        model = nltk.Model(self.valuation.domain, self.valuation)
        assingment = nltk.Assignment(self.valuation.domain)
        satisfier = model.satisfiers(expression, 'x3', assingment)
        return self.get_satisfier(self.valuation, satisfier)
