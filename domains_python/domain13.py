from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain13

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.52854, 'valueB': 0.8223, 'valueC': 0.03117, 'valueD': 0.0, 'valueE': 0.02587, 'valueF': 1.0},
        {'valueA': 0.59211, 'valueB': 0.27168, 'valueC': 0.16394, 'valueD': 0.45215, 'valueE': 1.0, 'valueF': 0.01674, 'valueG': 0.21837, 'valueH': 0.07186, 'valueI': 0.02723, 'valueJ': 0.15641, 'valueK': 0.56603, 'valueL': 0.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.42538, 'valueD': 0.29425, 'valueE': 0.33071},
        {'valueA': 0.06077, 'valueB': 0.3004, 'valueC': 0.0, 'valueD': 0.40252, 'valueE': 0.11339, 'valueF': 0.84993, 'valueG': 0.15895, 'valueH': 0.20608, 'valueI': 0.32973, 'valueJ': 0.34603, 'valueK': 0.21844, 'valueL': 0.32498, 'valueM': 0.11719, 'valueN': 0.1667, 'valueO': 0.08956, 'valueP': 0.01704, 'valueQ': 0.17389, 'valueR': 0.25415, 'valueS': 0.17226, 'valueT': 0.75955, 'valueU': 1.0, 'valueV': 0.06973, 'valueW': 0.0734, 'valueX': 0.21857, 'valueY': 0.28186, 'valueZ': 0.08406},
    ],
    weights=[
        0.03861,
        0.04158,
        0.33485,
        0.58496,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 0.26142, 'valueC': 0.34186, 'valueD': 0.47654, 'valueE': 0.42939, 'valueF': 1.0},
        {'valueA': 0.10063, 'valueB': 0.19143, 'valueC': 0.10117, 'valueD': 0.26312, 'valueE': 0.12133, 'valueF': 0.38293, 'valueG': 0.0, 'valueH': 0.34859, 'valueI': 0.36858, 'valueJ': 1.0, 'valueK': 0.25019, 'valueL': 0.14247},
        {'valueA': 0.58192, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.63195, 'valueE': 0.63148},
        {'valueA': 0.48307, 'valueB': 0.06322, 'valueC': 0.4327, 'valueD': 0.24784, 'valueE': 0.71086, 'valueF': 0.22661, 'valueG': 0.73811, 'valueH': 0.66473, 'valueI': 0.58947, 'valueJ': 0.0, 'valueK': 0.67358, 'valueL': 0.51433, 'valueM': 0.25468, 'valueN': 0.44166, 'valueO': 0.47162, 'valueP': 0.46171, 'valueQ': 0.40898, 'valueR': 0.06311, 'valueS': 1.0, 'valueT': 0.46809, 'valueU': 0.76772, 'valueV': 0.52176, 'valueW': 0.30128, 'valueX': 0.17848, 'valueY': 0.74283, 'valueZ': 0.23675},
    ],
    weights=[
        0.3718,
        0.07343,
        0.34537,
        0.2094,
    ],
    issues=issues,
    reserved_value=0.4,
)