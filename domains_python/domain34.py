from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain34

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.04069, 'valueB': 0.2965, 'valueC': 0.14659, 'valueD': 0.28777, 'valueE': 0.87993, 'valueF': 0.94488, 'valueG': 0.18865, 'valueH': 0.31354, 'valueI': 0.22774, 'valueJ': 0.39732, 'valueK': 0.2341, 'valueL': 0.10445, 'valueM': 0.11895, 'valueN': 0.99711, 'valueO': 0.22469, 'valueP': 0.2073, 'valueQ': 0.00839, 'valueR': 0.0, 'valueS': 0.00033, 'valueT': 0.46964, 'valueU': 0.21282, 'valueV': 1.0, 'valueW': 0.16152, 'valueX': 0.20722, 'valueY': 0.1321, 'valueZ': 0.08012},
        {'valueA': 0.77479, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 0.76536, 'valueB': 0.28731, 'valueC': 0.80565, 'valueD': 0.0, 'valueE': 1.0, 'valueF': 0.30413, 'valueG': 0.06002},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.07233, 'valueD': 0.32043, 'valueE': 0.1602, 'valueF': 0.25045},
    ],
    weights=[
        0.35435,
        0.46223,
        0.15786,
        0.02556,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.10643, 'valueB': 0.01085, 'valueC': 0.0, 'valueD': 0.0048, 'valueE': 0.16386, 'valueF': 0.68715, 'valueG': 0.21234, 'valueH': 0.16161, 'valueI': 0.07985, 'valueJ': 0.05126, 'valueK': 0.48543, 'valueL': 0.12085, 'valueM': 0.24342, 'valueN': 0.05636, 'valueO': 0.31035, 'valueP': 0.00901, 'valueQ': 0.0754, 'valueR': 0.25132, 'valueS': 0.14399, 'valueT': 0.05304, 'valueU': 1.0, 'valueV': 0.29611, 'valueW': 0.51023, 'valueX': 0.02432, 'valueY': 0.0579, 'valueZ': 0.30703},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.55862},
        {'valueA': 0.25402, 'valueB': 1.0, 'valueC': 0.02134, 'valueD': 0.00017, 'valueE': 0.06301, 'valueF': 0.0, 'valueG': 0.34085},
        {'valueA': 0.02557, 'valueB': 0.44979, 'valueC': 0.30032, 'valueD': 0.32026, 'valueE': 1.0, 'valueF': 0.0},
    ],
    weights=[
        0.34174,
        0.02118,
        0.46915,
        0.16793,
    ],
    issues=issues,
    reserved_value=0.4,
)