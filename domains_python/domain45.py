from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain45

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC', 'valueD']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.23375, 'valueB': 0.22663, 'valueC': 0.11703, 'valueD': 0.09556, 'valueE': 0.19835, 'valueF': 1.0, 'valueG': 0.0, 'valueH': 0.26464, 'valueI': 0.42423, 'valueJ': 0.16742, 'valueK': 0.09498, 'valueL': 0.05273, 'valueM': 0.55732},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.51491, 'valueD': 0.22785, 'valueE': 0.35463},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.20795},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.86668, 'valueD': 0.03153},
    ],
    weights=[
        0.00093,
        0.31077,
        0.2767,
        0.04905,
        0.10583,
        0.25672,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.39712, 'valueB': 1.0, 'valueC': 0.12795, 'valueD': 0.69913, 'valueE': 0.08151, 'valueF': 0.5482, 'valueG': 0.00468, 'valueH': 0.32408, 'valueI': 0.0, 'valueJ': 0.00034, 'valueK': 0.26623, 'valueL': 0.26378, 'valueM': 0.97209},
        {'valueA': 0.51055, 'valueB': 0.16841, 'valueC': 0.0, 'valueD': 0.0387, 'valueE': 1.0},
        {'valueA': 0.0, 'valueB': 0.31588, 'valueC': 1.0},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.02675, 'valueD': 0.18096},
    ],
    weights=[
        0.38921,
        0.00518,
        0.01327,
        0.16823,
        0.40462,
        0.01949,
    ],
    issues=issues,
    reserved_value=0.4,
)