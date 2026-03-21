from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain27

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.87176, 'valueB': 0.57546, 'valueC': 0.0, 'valueD': 0.35459, 'valueE': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.36665, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 0.53601, 'valueB': 0.04175, 'valueC': 0.1349, 'valueD': 0.05725, 'valueE': 0.65266, 'valueF': 0.29459, 'valueG': 0.38914, 'valueH': 0.57496, 'valueI': 1.0, 'valueJ': 0.28854, 'valueK': 0.09339, 'valueL': 0.10651, 'valueM': 0.21279, 'valueN': 0.0, 'valueO': 0.23397, 'valueP': 0.88969, 'valueQ': 0.33402},
        {'valueA': 0.60251, 'valueB': 0.0, 'valueC': 1.0},
    ],
    weights=[
        0.12613,
        0.10503,
        0.47546,
        0.12576,
        0.16762,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.32072, 'valueC': 0.33117, 'valueD': 0.0, 'valueE': 0.35606},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.50217, 'valueB': 1.0, 'valueC': 0.0},
        {'valueA': 0.15195, 'valueB': 0.41179, 'valueC': 1.0, 'valueD': 0.19646, 'valueE': 0.05485, 'valueF': 0.10874, 'valueG': 0.20379, 'valueH': 0.28343, 'valueI': 0.14441, 'valueJ': 0.29733, 'valueK': 0.69921, 'valueL': 0.41082, 'valueM': 0.44067, 'valueN': 0.03459, 'valueO': 0.0, 'valueP': 0.56159, 'valueQ': 0.82983},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.46259},
    ],
    weights=[
        0.14337,
        0.2555,
        0.22877,
        0.15513,
        0.21723,
    ],
    issues=issues,
    reserved_value=0.4,
)