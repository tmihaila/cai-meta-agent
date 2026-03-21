from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain15

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueE", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.03301, 'valueB': 0.29268, 'valueC': 0.39031, 'valueD': 0.04112, 'valueE': 0.25276, 'valueF': 1.0, 'valueG': 0.0, 'valueH': 0.03652},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.67004, 'valueC': 0.0, 'valueD': 0.84727, 'valueE': 0.11032, 'valueF': 0.06966, 'valueG': 0.85444, 'valueH': 0.09875},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.13867,
        0.06987,
        0.24028,
        0.25785,
        0.29333,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.05231, 'valueB': 0.0, 'valueC': 1.0, 'valueD': 0.27195, 'valueE': 0.20169, 'valueF': 0.11119, 'valueG': 0.2094, 'valueH': 0.04795},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.20026, 'valueC': 0.89371, 'valueD': 0.0, 'valueE': 0.46858, 'valueF': 0.18853, 'valueG': 0.48634, 'valueH': 0.04877},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.21739,
        0.39001,
        0.33929,
        0.00876,
        0.04455,
    ],
    issues=issues,
    reserved_value=0.4,
)