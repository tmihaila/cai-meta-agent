from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain20

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.51878, 'valueB': 0.41686, 'valueC': 0.0, 'valueD': 0.69827, 'valueE': 1.0},
        {'valueA': 0.24533, 'valueB': 0.2578, 'valueC': 0.00085, 'valueD': 0.0, 'valueE': 1.0, 'valueF': 0.57497, 'valueG': 0.33815, 'valueH': 0.115},
        {'valueA': 0.49296, 'valueB': 0.1972, 'valueC': 0.0, 'valueD': 1.0},
        {'valueA': 0.0189, 'valueB': 0.12895, 'valueC': 1.0, 'valueD': 0.33591, 'valueE': 0.06014, 'valueF': 0.35933, 'valueG': 0.30035, 'valueH': 0.20513, 'valueI': 0.13952, 'valueJ': 0.12277, 'valueK': 0.44072, 'valueL': 0.12592, 'valueM': 0.0},
    ],
    weights=[
        0.26901,
        0.16438,
        0.26157,
        0.30504,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.10289, 'valueB': 0.0218, 'valueC': 1.0, 'valueD': 0.22418, 'valueE': 0.0},
        {'valueA': 0.08602, 'valueB': 1.0, 'valueC': 0.06293, 'valueD': 0.58767, 'valueE': 0.0, 'valueF': 0.16144, 'valueG': 0.20481, 'valueH': 0.17869},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.82352, 'valueD': 0.45872},
        {'valueA': 0.21087, 'valueB': 0.01219, 'valueC': 0.16177, 'valueD': 0.0455, 'valueE': 0.39926, 'valueF': 0.06306, 'valueG': 0.09463, 'valueH': 0.06686, 'valueI': 1.0, 'valueJ': 0.62202, 'valueK': 0.43498, 'valueL': 0.0, 'valueM': 0.29191},
    ],
    weights=[
        0.05984,
        0.29949,
        0.46173,
        0.17894,
    ],
    issues=issues,
    reserved_value=0.4,
)