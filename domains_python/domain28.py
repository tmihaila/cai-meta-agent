from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain28

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.44847, 'valueB': 0.0, 'valueC': 0.1839, 'valueD': 1.0},
        {'valueA': 0.0, 'valueB': 0.32857, 'valueC': 0.39336, 'valueD': 0.95113, 'valueE': 1.0, 'valueF': 0.58553},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.76544, 'valueD': 0.52501},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.05055, 'valueB': 0.28856, 'valueC': 0.2059, 'valueD': 0.27551, 'valueE': 0.02912, 'valueF': 0.06954, 'valueG': 0.0, 'valueH': 0.21426, 'valueI': 1.0, 'valueJ': 0.83253},
        {'valueA': 0.4074, 'valueB': 1.0, 'valueC': 0.36903, 'valueD': 0.21099, 'valueE': 0.0},
    ],
    weights=[
        0.02423,
        0.09786,
        0.21459,
        0.55512,
        0.07699,
        0.03121,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.48317, 'valueB': 0.78112, 'valueC': 1.0, 'valueD': 0.0},
        {'valueA': 0.34677, 'valueB': 0.0, 'valueC': 0.06851, 'valueD': 1.0, 'valueE': 0.13308, 'valueF': 0.08424},
        {'valueA': 0.13449, 'valueB': 0.0, 'valueC': 0.20992, 'valueD': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.10151, 'valueC': 0.68372, 'valueD': 0.13838, 'valueE': 0.23901, 'valueF': 0.02876, 'valueG': 0.50397, 'valueH': 0.09499, 'valueI': 0.2906, 'valueJ': 0.0},
        {'valueA': 0.22517, 'valueB': 1.0, 'valueC': 0.0386, 'valueD': 0.0, 'valueE': 0.24516},
    ],
    weights=[
        0.44001,
        0.06338,
        0.15223,
        0.06288,
        0.09782,
        0.18368,
    ],
    issues=issues,
    reserved_value=0.4,
)