from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain10

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueE", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.11322, 'valueB': 0.00467, 'valueC': 0.30247, 'valueD': 0.60489, 'valueE': 0.13324, 'valueF': 1.0, 'valueG': 0.44396, 'valueH': 0.65664, 'valueI': 0.1522, 'valueJ': 0.32312, 'valueK': 0.53806, 'valueL': 0.08311, 'valueM': 0.0, 'valueN': 0.29385, 'valueO': 0.20538, 'valueP': 0.07592, 'valueQ': 0.18536},
        {'valueA': 0.57777, 'valueB': 0.0, 'valueC': 0.18967, 'valueD': 1.0},
        {'valueA': 1.0, 'valueB': 0.2508, 'valueC': 0.0, 'valueD': 0.26437, 'valueE': 0.63207},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.37769},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.50494,
        0.21606,
        0.1918,
        0.07639,
        0.01081,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.76223, 'valueB': 0.12063, 'valueC': 0.59609, 'valueD': 0.06709, 'valueE': 0.07499, 'valueF': 0.52284, 'valueG': 0.04226, 'valueH': 0.9156, 'valueI': 0.48233, 'valueJ': 0.28534, 'valueK': 0.02489, 'valueL': 0.18791, 'valueM': 0.0, 'valueN': 0.15796, 'valueO': 0.39437, 'valueP': 1.0, 'valueQ': 0.77759},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.98985, 'valueD': 0.00578},
        {'valueA': 1.0, 'valueB': 0.60046, 'valueC': 0.0, 'valueD': 0.09683, 'valueE': 0.12858},
        {'valueA': 0.08618, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.00975,
        0.02423,
        0.26318,
        0.28194,
        0.4209,
    ],
    issues=issues,
    reserved_value=0.4,
)