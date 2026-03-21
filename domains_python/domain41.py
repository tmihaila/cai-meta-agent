from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain41

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.75884, 'valueD': 0.66178},
        {'valueA': 0.40751, 'valueB': 0.24242, 'valueC': 0.05707, 'valueD': 0.76754, 'valueE': 0.0, 'valueF': 0.30152, 'valueG': 0.03292, 'valueH': 1.0, 'valueI': 0.35031},
        {'valueA': 0.07468, 'valueB': 0.13282, 'valueC': 0.0, 'valueD': 0.07935, 'valueE': 0.09641, 'valueF': 0.47309, 'valueG': 1.0, 'valueH': 0.17997},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 0.77909, 'valueC': 1.0, 'valueD': 0.31853},
        {'valueA': 0.98367, 'valueB': 1.0, 'valueC': 0.0},
    ],
    weights=[
        0.07647,
        0.3324,
        0.40647,
        0.02662,
        0.01897,
        0.13907,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 0.76905, 'valueC': 1.0, 'valueD': 0.89997},
        {'valueA': 0.37028, 'valueB': 1.0, 'valueC': 0.06884, 'valueD': 0.07039, 'valueE': 0.0, 'valueF': 0.41638, 'valueG': 0.41572, 'valueH': 0.05475, 'valueI': 0.89594},
        {'valueA': 0.18469, 'valueB': 0.47821, 'valueC': 0.16128, 'valueD': 1.0, 'valueE': 0.13333, 'valueF': 0.0, 'valueG': 0.61538, 'valueH': 0.05343},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.94269, 'valueD': 0.04549},
        {'valueA': 0.27531, 'valueB': 1.0, 'valueC': 0.0},
    ],
    weights=[
        0.11594,
        0.12745,
        0.32802,
        0.23207,
        0.11567,
        0.08085,
    ],
    issues=issues,
    reserved_value=0.4,
)