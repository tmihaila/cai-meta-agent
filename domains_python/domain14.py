from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain14

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueG", values=['valueA', 'valueB', 'valueC']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.22981, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.07587},
        {'valueA': 1.0, 'valueB': 0.15463, 'valueC': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.55593, 'valueB': 0.0, 'valueC': 0.31963, 'valueD': 1.0, 'valueE': 0.10432},
        {'valueA': 0.17635, 'valueB': 0.0, 'valueC': 1.0, 'valueD': 0.1054, 'valueE': 0.68067},
        {'valueA': 1.0, 'valueB': 0.51695, 'valueC': 0.0},
    ],
    weights=[
        0.07881,
        0.06857,
        0.06612,
        0.31052,
        0.16364,
        0.02764,
        0.2847,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.82367, 'valueB': 0.0, 'valueC': 1.0, 'valueD': 0.32019},
        {'valueA': 0.25286, 'valueB': 1.0, 'valueC': 0.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.30346, 'valueC': 0.78546, 'valueD': 0.89067, 'valueE': 0.0},
        {'valueA': 0.71158, 'valueB': 0.0, 'valueC': 0.97849, 'valueD': 0.25148, 'valueE': 1.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.80326},
    ],
    weights=[
        0.01281,
        0.37395,
        0.12845,
        0.00638,
        0.03704,
        0.42248,
        0.01889,
    ],
    issues=issues,
    reserved_value=0.4,
)