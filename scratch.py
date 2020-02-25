from sklearn.feature_extraction.text import CountVectorizer
import nltk

vectoriser = CountVectorizer()

corpus = [
    'I have played x',
    'Have I played X',
    'I liked X',
    'I did not like X',
    'What games do I like',
    'What games did I not like/ What games don\'t I like',
    'What games have I played',
    'What games have I played and liked',
    'What games have I played and not liked',
]

read_expr = nltk.sem.Expression.fromstring

initial_expression = r'\y.exists x.(game(x) & like(y,x))' 
expr = read_expr(initial_expression)

trans_verb_phrase = read_expr(r'\X x.X(\y.like(x,y))')
noun_phrase = read_expr(r'(\P.exists x.(game(x) & P(x)))')
vp = nltk.sem.ApplicationExpression(trans_verb_phrase, noun_phrase)
print(vp)
print(vp.simplify())