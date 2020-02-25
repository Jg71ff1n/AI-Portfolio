from nltk import load_parser
import nltk
read_expr = nltk.sem.Expression.fromstring

phrase = 'joe likes chess'

# Lexical Rules
initial_rules = r'''
    PropN[-LOC, NUM=sg, SEM=<\P.P(joe)>] -> 'joe' # P.P(name) is a more abstract way to represent names
    TV[NUM=sg, SEM=? TNS=pres] -> 'likes' # Need to fill in the semantics
    DET[NUM=sg, SEM=?] -> 'the' # Need to fill in the semantics
    N[NUM=sg, SEM=<\x.game(x)>] -> 'game' # Descriptor semantic meaning
    N[NUM=sg, SEM=<\x.chess(x)>] -> 'chess' # Descriptor semantic meaning
'''

# Turn our phrase into generalised logic
initial_expression = r'\x y.(game(y) & like(x,y))(Joe, Chess)'

# Turn the string into a NLTK expression
expr = read_expr(initial_expression)
# print(expr) # Print the raw expression
# print(expr.simplify()) # Simplify it and put our labels into the expression

# Need to abstract x away -> introduce P to replace x Chapter 10-4.3 of NLTK book
generalised_noun_phrase = read_expr(r'(\P.exists y.(game(y) & P(y)))')
# We still have the verb phrase left
left_over_verb_phrase = r'\z.like(x,y)'

# Need to replace function with a variable, in this case X, of the same type as the noun phrase
replaced_verb_phrase = r'X(\z.like(x,y))'

# Abstract out X and y to give:
reduced_trans_verb_phrase = read_expr(r'\X x.X(\y.like(x,y))')

# Combine them together
vp = nltk.sem.ApplicationExpression(
    reduced_trans_verb_phrase, generalised_noun_phrase)
# print(vp) # Print the complex expression
# print(vp.simplify()) # Simplify it, and we get the expression at the start, minus the people in it

# Lexical Rules
final_rules = r'''
    PropN[-LOC, NUM=sg, SEM=<\P.P(joe)>] -> 'joe' # P.P(name) is a more abstract way to represent names
    TV[NUM=sg, SEM=<\X x.X(\y.like(x,y))> TNS=pres] -> 'likes' # From reduced_trans_verb_phrase
    DET[NUM=sg, SEM=<\P.exists y.(game(y) & P(y))>] -> 'the' # From generalised noun phrase
    N[NUM=sg, SEM=<\x.game(x)>] -> 'game' # Descriptor semantic meaning
    N[NUM=sg, SEM=<\x.chess(x)>] -> 'chess' # Descriptor semantic meaning
'''


parser = load_parser('board_games.fcfg', trace=0)
tokens = 'King is north of the Pawn'.split()
# tokens = phrase.split()
x = parser.parse(tokens)
for tree in x:
    semantic = tree.label()['SEM']
    print(semantic)

# for results in nltk.interpret_sents(['joe is above jack'], 'board_games.fcfg'):
#     for (synrep, semrep) in results:
#         print(synrep)

v = """
    bertie => b
    olive => o
    cyril => c
    chess => ch
    joe => jo
    jack => ja
    boy => {b}
    girl => {o}
    dog => {c}
    walk => {o, c}
    see => {(b, o), (c, b), (o, c)}
    north => {(jo,ja)}
    south => {(jo,ja)}
    like => {(jo,ch)}
"""
# a6 = read_expr('north(john,joe)')
# ns_goals = read_expr('south(joe, jack)')
# all_below_y_goal = read_expr('all x.(south(x,joe))')
# all_above_y_goal = read_expr('all x.(north(x,joe))')


# mb = nltk.Mace(5)
# a1 = read_expr('north(x,y) -> south(y,x)')
# a2 = read_expr('north(x,y) & north(y,z) -> north(x,z)')
# a3 = read_expr('bpawn(joe)')
# a4 = read_expr('wpawn(jack)')
# a5 = read_expr('north(joe,jack)')
# a6 = read_expr('all x.(bpawn(x) -> -wpawn(x))')

# anything_above_jack_goal = read_expr('exists x.(north(x,jack))')
# anything_below_jack_goal = read_expr('exists x.(south(x,jack))')

# prover = nltk.Prover9()
# print(prover.prove(anything_above_jack_goal, [a1, a2, a3, a4, a5, a6]))

# mc = nltk.MaceCommand(None, assumptions=[a1, a2, a3, a4, a5, a6])
# print(mc.build_model())
# print(mc.valuation)

# # What is above y
# y = 'joe'

# dom2 = mc.valuation.domain
# m2 = nltk.Model(dom2, mc.valuation)
# g2 = nltk.Assignment(dom2)
# who_is_above_y = read_expr(f'north(x,{y})')
# who_is_below_y = read_expr(f'north({y},x)')
# satisfier = m2.satisfiers(who_is_below_y, 'x', g2)
# print(satisfier)


# text = nltk.word_tokenize('is there anything above Pawn')
# tagged = nltk.pos_tag(text)
# print(tagged)

# for word, wordClass in tagged:
#     print(nltk.help.upenn_tagset(wordClass))


# a4 = read_expr('exists y. (woman(y) & all x. (man(x) -> love(x,y)))')
# a5 = read_expr('man(adam)')
# a6 = read_expr('woman(eve)')
# g = read_expr('love(adam,eve)')
# mc = nltk.MaceCommand(g, assumptions=[a4, a5, a6])
# print(mc.build_model())
