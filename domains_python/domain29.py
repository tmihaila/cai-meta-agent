from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain29

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC', 'valueD']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.19346, 'valueB': 0.0, 'valueC': 1.0, 'valueD': 0.10494, 'valueE': 0.11748},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.27546, 'valueD': 0.03354, 'valueE': 0.22879, 'valueF': 0.30241, 'valueG': 0.93327, 'valueH': 0.49123, 'valueI': 0.03433},
        {'valueA': 0.89907, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.28447},
        {'valueA': 0.0, 'valueB': 0.306, 'valueC': 1.0, 'valueD': 0.04802, 'valueE': 0.02104},
        {'valueA': 0.94831, 'valueB': 1.0, 'valueC': 0.25352, 'valueD': 0.0},
    ],
    weights=[
        0.13384,
        0.04335,
        0.08472,
        0.19306,
        0.3774,
        0.16763,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.16429, 'valueB': 0.53536, 'valueC': 0.0, 'valueD': 0.1426, 'valueE': 1.0},
        {'valueA': 0.42976, 'valueB': 0.52465, 'valueC': 0.05499, 'valueD': 0.07624, 'valueE': 0.1009, 'valueF': 0.00538, 'valueG': 1.0, 'valueH': 0.0, 'valueI': 0.30551},
        {'valueA': 0.49031, 'valueB': 1.0, 'valueC': 0.0},
        {'valueA': 0.0, 'valueB': 0.09115, 'valueC': 1.0},
        {'valueA': 1.0, 'valueB': 0.8338, 'valueC': 0.92687, 'valueD': 0.0, 'valueE': 0.2072},
        {'valueA': 0.07705, 'valueB': 0.37945, 'valueC': 1.0, 'valueD': 0.0},
    ],
    weights=[
        0.01776,
        0.13457,
        0.16324,
        0.07094,
        0.20581,
        0.40768,
    ],
    issues=issues,
    reserved_value=0.4,
)