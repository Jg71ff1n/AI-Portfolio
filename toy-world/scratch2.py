from nltk import load_parser
import nltk
read_expr = nltk.sem.Expression.fromstring


def get_satisfier(valuation: nltk.Valuation, satisfiers: set):
    results = []
    for satisfier in satisfiers:
        for k, v in valuation.items():
            print(f'key = {k} value = {v}')
            if len(v) == 1:
                if v == satisfier:
                    results.append(k)
    return results


mb = nltk.Mace(5)
a1 = read_expr('north(x,y) <-> south(y,x)')
a2 = read_expr('north(x,y) & north(y,z) -> north(x,z)')
a3 = read_expr('all x.(bpawn(x) -> -wpawn(x))')
a4 = read_expr('Pawn != Rook')
a5 = read_expr('Pawn != King')
a6 = read_expr('Rook != King')
a7 = read_expr('north(Pawn,Rook)')
a8 = read_expr('south(Pawn,King)')

y = 'King'

anything_above_y_goal = read_expr(f'exists x.north(x,{y})')
anything_below_y_goal = read_expr(f'exists x.south(x,{y})')

prover = nltk.Prover9()
print(prover.prove(anything_above_y_goal, [a1, a2, a3, a4, a5, a6, a7, a8]))

mc = nltk.MaceCommand(None, assumptions=[a1, a2, a3, a4, a5, a6, a7, a8])
# mc.print_assumptions()
print(mc.build_model())
# print(mc.valuation)

dom2 = mc.valuation.domain
m2 = nltk.Model(dom2, mc.valuation)
g2 = nltk.Assignment(dom2)
who_is_above_y = read_expr(f'north(x,{y})')
who_is_below_y = read_expr(f'south(x,{y})')
satisfier = m2.satisfiers(who_is_below_y, 'x', g2)
# print(get_satisfier(mc.valuation, satisfier))
