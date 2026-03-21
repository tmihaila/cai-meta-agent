from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain48

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.077, 'valueB': 0.1831, 'valueC': 0.0, 'valueD': 0.26367, 'valueE': 0.5245, 'valueF': 0.13571, 'valueG': 1.0},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.23364, 'valueD': 0.17036, 'valueE': 0.16341, 'valueF': 0.5493, 'valueG': 0.244, 'valueH': 0.01533, 'valueI': 0.19616, 'valueJ': 0.19716, 'valueK': 0.3204, 'valueL': 0.05219},
        {'valueA': 0.16341, 'valueB': 0.6184, 'valueC': 0.0, 'valueD': 0.02303, 'valueE': 1.0, 'valueF': 0.26868},
        {'valueA': 0.03197, 'valueB': 0.00176, 'valueC': 0.21245, 'valueD': 0.0, 'valueE': 1.0, 'valueF': 0.00602, 'valueG': 0.54031, 'valueH': 0.19321, 'valueI': 0.09866, 'valueJ': 0.11762, 'valueK': 0.02612, 'valueL': 0.15278, 'valueM': 0.61299, 'valueN': 0.03169, 'valueO': 0.16171},
    ],
    weights=[
        0.50832,
        0.20078,
        0.00524,
        0.28566,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.08133, 'valueB': 0.08196, 'valueC': 0.01285, 'valueD': 0.36845, 'valueE': 0.14342, 'valueF': 1.0, 'valueG': 0.0},
        {'valueA': 1.0, 'valueB': 0.51826, 'valueC': 0.0, 'valueD': 0.12404, 'valueE': 0.73345, 'valueF': 0.83016, 'valueG': 0.3089, 'valueH': 0.15945, 'valueI': 0.17175, 'valueJ': 0.23091, 'valueK': 0.74982, 'valueL': 0.3326},
        {'valueA': 0.01385, 'valueB': 1.0, 'valueC': 0.15081, 'valueD': 0.0, 'valueE': 0.01698, 'valueF': 0.41214},
        {'valueA': 0.31426, 'valueB': 0.09616, 'valueC': 0.02711, 'valueD': 1.0, 'valueE': 0.16444, 'valueF': 0.45859, 'valueG': 0.1689, 'valueH': 0.05189, 'valueI': 0.21937, 'valueJ': 0.23079, 'valueK': 0.05127, 'valueL': 0.28441, 'valueM': 0.0, 'valueN': 0.24133, 'valueO': 0.4742},
    ],
    weights=[
        0.29201,
        0.05416,
        0.21338,
        0.44045,
    ],
    issues=issues,
    reserved_value=0.4,
)