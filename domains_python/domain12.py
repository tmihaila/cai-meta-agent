from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain12

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC', 'valueD']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.83089, 'valueB': 0.02367, 'valueC': 1.0, 'valueD': 0.03103, 'valueE': 0.07632, 'valueF': 0.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.14014, 'valueD': 0.16006, 'valueE': 0.94664, 'valueF': 0.48846, 'valueG': 0.06175},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.44397, 'valueC': 0.0, 'valueD': 0.30446},
    ],
    weights=[
        0.12058,
        0.01364,
        0.11678,
        0.03102,
        0.43683,
        0.28115,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.12703, 'valueB': 0.05106, 'valueC': 0.43186, 'valueD': 0.0, 'valueE': 1.0, 'valueF': 0.36014},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.60562, 'valueB': 0.2719, 'valueC': 0.0, 'valueD': 1.0, 'valueE': 0.68411, 'valueF': 0.14092, 'valueG': 0.27343},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.0038, 'valueC': 0.38396, 'valueD': 0.0},
    ],
    weights=[
        0.15375,
        0.42023,
        0.03166,
        0.11067,
        0.17522,
        0.10847,
    ],
    issues=issues,
    reserved_value=0.4,
)