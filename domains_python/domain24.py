from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain24

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.10077, 'valueB': 0.18942, 'valueC': 0.8868, 'valueD': 0.36248, 'valueE': 0.17725, 'valueF': 0.09718, 'valueG': 1.0, 'valueH': 0.7757, 'valueI': 0.01202, 'valueJ': 0.60419, 'valueK': 0.0, 'valueL': 0.45757},
        {'valueA': 0.08495, 'valueB': 0.03366, 'valueC': 0.12635, 'valueD': 0.39596, 'valueE': 0.05954, 'valueF': 0.17924, 'valueG': 1.0, 'valueH': 0.0},
        {'valueA': 0.09983, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.30125, 'valueE': 0.29318},
        {'valueA': 0.17987, 'valueB': 0.3896, 'valueC': 0.86424, 'valueD': 0.00064, 'valueE': 0.0, 'valueF': 0.26312, 'valueG': 0.21221, 'valueH': 0.57882, 'valueI': 0.1631, 'valueJ': 0.16009, 'valueK': 1.0, 'valueL': 0.1629, 'valueM': 0.87724, 'valueN': 0.17462, 'valueO': 0.19608, 'valueP': 0.31992, 'valueQ': 0.20639},
    ],
    weights=[
        0.00527,
        0.11534,
        0.29642,
        0.58297,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.13527, 'valueB': 0.08246, 'valueC': 0.0, 'valueD': 1.0, 'valueE': 0.42323, 'valueF': 0.21933, 'valueG': 0.39507, 'valueH': 0.1432, 'valueI': 0.3508, 'valueJ': 0.72105, 'valueK': 0.00375, 'valueL': 0.82318},
        {'valueA': 0.12755, 'valueB': 0.20441, 'valueC': 0.27721, 'valueD': 0.02102, 'valueE': 1.0, 'valueF': 0.0, 'valueG': 0.62472, 'valueH': 0.18384},
        {'valueA': 0.10627, 'valueB': 1.0, 'valueC': 0.01221, 'valueD': 0.35231, 'valueE': 0.0},
        {'valueA': 0.17627, 'valueB': 0.40483, 'valueC': 0.55919, 'valueD': 0.04866, 'valueE': 0.39652, 'valueF': 1.0, 'valueG': 0.11854, 'valueH': 0.34105, 'valueI': 0.09845, 'valueJ': 0.0, 'valueK': 0.53353, 'valueL': 0.45933, 'valueM': 0.13302, 'valueN': 0.2527, 'valueO': 0.27759, 'valueP': 0.29261, 'valueQ': 0.03159},
    ],
    weights=[
        0.11655,
        0.24342,
        0.1757,
        0.46433,
    ],
    issues=issues,
    reserved_value=0.4,
)