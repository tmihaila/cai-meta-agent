from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain38

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.07189, 'valueB': 0.15716, 'valueC': 0.19711, 'valueD': 0.16438, 'valueE': 0.1739, 'valueF': 0.08187, 'valueG': 1.0, 'valueH': 0.08538, 'valueI': 0.44317, 'valueJ': 0.0, 'valueK': 0.01064, 'valueL': 0.55507, 'valueM': 0.60108, 'valueN': 0.06413, 'valueO': 0.30985, 'valueP': 0.00242, 'valueQ': 0.04672},
        {'valueA': 0.83105, 'valueB': 0.0, 'valueC': 0.15903, 'valueD': 0.28429, 'valueE': 1.0},
        {'valueA': 0.29108, 'valueB': 0.2113, 'valueC': 0.0, 'valueD': 0.8681, 'valueE': 0.82029, 'valueF': 1.0},
        {'valueA': 0.78016, 'valueB': 0.56727, 'valueC': 1.0, 'valueD': 0.0},
    ],
    weights=[
        0.12449,
        0.15379,
        0.13003,
        0.59169,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.4161, 'valueB': 0.24836, 'valueC': 0.23938, 'valueD': 0.33527, 'valueE': 1.0, 'valueF': 0.41086, 'valueG': 0.29391, 'valueH': 0.02988, 'valueI': 0.39704, 'valueJ': 0.22521, 'valueK': 0.24761, 'valueL': 0.28099, 'valueM': 0.15252, 'valueN': 0.08771, 'valueO': 0.0, 'valueP': 0.35348, 'valueQ': 0.13391},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.64735, 'valueD': 0.74462, 'valueE': 0.97988},
        {'valueA': 0.22364, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.67592, 'valueE': 0.26234, 'valueF': 0.09184},
        {'valueA': 0.41788, 'valueB': 1.0, 'valueC': 0.66761, 'valueD': 0.0},
    ],
    weights=[
        0.15584,
        0.00436,
        0.1447,
        0.6951,
    ],
    issues=issues,
    reserved_value=0.4,
)