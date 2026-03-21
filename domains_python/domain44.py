from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain44

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK']),
    make_issue(name="issueF", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.63205},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.10282, 'valueB': 0.09074, 'valueC': 0.00167, 'valueD': 0.17841, 'valueE': 1.0, 'valueF': 0.10492, 'valueG': 0.1094, 'valueH': 0.43436, 'valueI': 0.0, 'valueJ': 0.50914, 'valueK': 0.33144},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.38003,
        0.16935,
        0.00961,
        0.11727,
        0.05994,
        0.2638,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.106},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.13599, 'valueC': 0.04515, 'valueD': 0.76991, 'valueE': 0.0, 'valueF': 0.83632, 'valueG': 0.3341, 'valueH': 0.38856, 'valueI': 0.25668, 'valueJ': 0.16972, 'valueK': 0.74123},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.03399,
        0.28419,
        0.41278,
        0.21081,
        0.04964,
        0.00859,
    ],
    issues=issues,
    reserved_value=0.4,
)