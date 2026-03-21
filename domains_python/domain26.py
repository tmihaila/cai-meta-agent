from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain26

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.04819, 'valueB': 0.7631, 'valueC': 0.0, 'valueD': 0.09411, 'valueE': 1.0, 'valueF': 0.04866, 'valueG': 0.40725, 'valueH': 0.05524, 'valueI': 0.4737, 'valueJ': 0.25977, 'valueK': 0.21926, 'valueL': 0.30381, 'valueM': 0.32622, 'valueN': 0.40741, 'valueO': 0.00297, 'valueP': 0.51524, 'valueQ': 0.09105, 'valueR': 0.0344, 'valueS': 0.3252, 'valueT': 0.03134, 'valueU': 0.17937, 'valueV': 0.09395, 'valueW': 0.5092, 'valueX': 0.91983},
        {'valueA': 1.0, 'valueB': 0.16644, 'valueC': 0.62346, 'valueD': 0.0, 'valueE': 0.15966},
        {'valueA': 0.51971, 'valueB': 0.0006, 'valueC': 0.31252, 'valueD': 0.03883, 'valueE': 0.11148, 'valueF': 0.65135, 'valueG': 0.2136, 'valueH': 0.02575, 'valueI': 0.82878, 'valueJ': 0.79791, 'valueK': 0.0, 'valueL': 1.0, 'valueM': 0.04344},
        {'valueA': 0.49728, 'valueB': 0.54304, 'valueC': 1.0, 'valueD': 0.63484, 'valueE': 0.0, 'valueF': 0.45427},
    ],
    weights=[
        0.58367,
        0.19173,
        0.01155,
        0.21305,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.07423, 'valueB': 0.27644, 'valueC': 0.43262, 'valueD': 0.13552, 'valueE': 0.0, 'valueF': 0.3193, 'valueG': 0.31567, 'valueH': 0.28692, 'valueI': 0.10846, 'valueJ': 1.0, 'valueK': 0.46045, 'valueL': 0.32377, 'valueM': 0.15009, 'valueN': 0.22733, 'valueO': 0.06683, 'valueP': 0.05103, 'valueQ': 0.14037, 'valueR': 0.92175, 'valueS': 0.77705, 'valueT': 0.15965, 'valueU': 0.79502, 'valueV': 0.01295, 'valueW': 0.29363, 'valueX': 0.19588},
        {'valueA': 0.22642, 'valueB': 0.43187, 'valueC': 1.0, 'valueD': 0.60115, 'valueE': 0.0},
        {'valueA': 0.32409, 'valueB': 0.0, 'valueC': 0.63779, 'valueD': 0.25927, 'valueE': 0.9952, 'valueF': 0.58503, 'valueG': 1.0, 'valueH': 0.43588, 'valueI': 0.89894, 'valueJ': 0.78016, 'valueK': 0.27479, 'valueL': 0.17707, 'valueM': 0.16487},
        {'valueA': 0.6287, 'valueB': 1.0, 'valueC': 0.84385, 'valueD': 0.0, 'valueE': 0.38397, 'valueF': 0.52277},
    ],
    weights=[
        0.31464,
        0.15039,
        0.24874,
        0.28623,
    ],
    issues=issues,
    reserved_value=0.4,
)