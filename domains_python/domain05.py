from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain05

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.91748, 'valueB': 0.26856, 'valueC': 0.38091, 'valueD': 0.0, 'valueE': 0.75045, 'valueF': 0.73704, 'valueG': 0.71363, 'valueH': 1.0},
        {'valueA': 0.00327, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.33737},
        {'valueA': 0.0, 'valueB': 0.06194, 'valueC': 0.30947, 'valueD': 0.55008, 'valueE': 0.30092, 'valueF': 0.47391, 'valueG': 0.38942, 'valueH': 0.38837, 'valueI': 0.51888, 'valueJ': 0.50796, 'valueK': 0.04509, 'valueL': 0.61638, 'valueM': 1.0, 'valueN': 0.62661},
        {'valueA': 0.18622, 'valueB': 0.0, 'valueC': 0.71202, 'valueD': 0.40707, 'valueE': 0.16647, 'valueF': 1.0, 'valueG': 0.7162},
    ],
    weights=[
        0.09534,
        0.00898,
        0.59264,
        0.30304,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.39527, 'valueB': 0.02192, 'valueC': 0.0, 'valueD': 0.54815, 'valueE': 0.45858, 'valueF': 0.35392, 'valueG': 0.83848, 'valueH': 1.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.11197, 'valueD': 0.69145},
        {'valueA': 0.22581, 'valueB': 0.0, 'valueC': 0.20079, 'valueD': 0.10851, 'valueE': 0.04181, 'valueF': 0.16691, 'valueG': 1.0, 'valueH': 0.66107, 'valueI': 0.12505, 'valueJ': 0.10553, 'valueK': 0.37488, 'valueL': 0.38705, 'valueM': 0.02868, 'valueN': 0.08668},
        {'valueA': 0.76385, 'valueB': 1.0, 'valueC': 0.26575, 'valueD': 0.25059, 'valueE': 0.2902, 'valueF': 0.09215, 'valueG': 0.0},
    ],
    weights=[
        0.36941,
        0.09076,
        0.27795,
        0.26188,
    ],
    issues=issues,
    reserved_value=0.4,
)