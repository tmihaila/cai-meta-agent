from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain08

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.5402, 'valueC': 0.86269, 'valueD': 0.0, 'valueE': 0.04184},
        {'valueA': 0.33029, 'valueB': 0.11178, 'valueC': 0.00104, 'valueD': 0.05576, 'valueE': 0.41586, 'valueF': 0.0, 'valueG': 0.30465, 'valueH': 0.24339, 'valueI': 0.20612, 'valueJ': 0.04959, 'valueK': 1.0},
        {'valueA': 0.01108, 'valueB': 0.13428, 'valueC': 0.29939, 'valueD': 0.78056, 'valueE': 0.29536, 'valueF': 0.55926, 'valueG': 1.0, 'valueH': 0.0},
        {'valueA': 0.06062, 'valueB': 0.30958, 'valueC': 0.29896, 'valueD': 0.0, 'valueE': 0.50594, 'valueF': 1.0, 'valueG': 0.37615, 'valueH': 0.39148, 'valueI': 0.61405},
    ],
    weights=[
        0.32499,
        0.05273,
        0.29743,
        0.32485,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.98015, 'valueB': 0.8902, 'valueC': 1.0, 'valueD': 0.41422, 'valueE': 0.0},
        {'valueA': 0.30008, 'valueB': 0.35544, 'valueC': 0.07636, 'valueD': 0.3391, 'valueE': 0.35653, 'valueF': 0.13654, 'valueG': 1.0, 'valueH': 0.28717, 'valueI': 0.00259, 'valueJ': 0.0, 'valueK': 0.85662},
        {'valueA': 0.78717, 'valueB': 0.0, 'valueC': 0.1263, 'valueD': 0.31593, 'valueE': 1.0, 'valueF': 0.34435, 'valueG': 0.32358, 'valueH': 0.21791},
        {'valueA': 0.0, 'valueB': 0.55978, 'valueC': 0.76737, 'valueD': 0.06008, 'valueE': 0.075, 'valueF': 0.12449, 'valueG': 1.0, 'valueH': 0.34507, 'valueI': 0.20446},
    ],
    weights=[
        0.47267,
        0.07305,
        0.21224,
        0.24204,
    ],
    issues=issues,
    reserved_value=0.4,
)