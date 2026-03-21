from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain19

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.04658, 'valueC': 0.51781, 'valueD': 0.0525, 'valueE': 0.04139, 'valueF': 0.65647, 'valueG': 0.0, 'valueH': 0.42422},
        {'valueA': 0.44035, 'valueB': 0.35877, 'valueC': 0.48929, 'valueD': 0.63185, 'valueE': 0.01591, 'valueF': 0.34569, 'valueG': 0.7249, 'valueH': 0.54583, 'valueI': 0.25531, 'valueJ': 0.12664, 'valueK': 0.28858, 'valueL': 0.13302, 'valueM': 0.45585, 'valueN': 0.15386, 'valueO': 0.0, 'valueP': 0.0781, 'valueQ': 0.16379, 'valueR': 0.34302, 'valueS': 0.30781, 'valueT': 0.45085, 'valueU': 0.17906, 'valueV': 1.0, 'valueW': 0.04385, 'valueX': 0.08084, 'valueY': 0.03715, 'valueZ': 0.02859},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.48862, 'valueB': 0.03824, 'valueC': 0.26031, 'valueD': 0.28735, 'valueE': 0.0, 'valueF': 0.70967, 'valueG': 1.0, 'valueH': 0.16356, 'valueI': 0.7855},
    ],
    weights=[
        0.02044,
        0.73263,
        0.24343,
        0.0035,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.08188, 'valueB': 0.24059, 'valueC': 0.02334, 'valueD': 0.40009, 'valueE': 0.27941, 'valueF': 0.25532, 'valueG': 1.0, 'valueH': 0.0},
        {'valueA': 0.37514, 'valueB': 0.04789, 'valueC': 0.10343, 'valueD': 0.06898, 'valueE': 0.25563, 'valueF': 0.35544, 'valueG': 0.4434, 'valueH': 0.1288, 'valueI': 0.61332, 'valueJ': 0.0166, 'valueK': 0.11543, 'valueL': 0.17775, 'valueM': 0.35841, 'valueN': 0.05125, 'valueO': 0.02813, 'valueP': 0.03452, 'valueQ': 0.10699, 'valueR': 0.38391, 'valueS': 0.17802, 'valueT': 1.0, 'valueU': 0.0, 'valueV': 0.15878, 'valueW': 0.37172, 'valueX': 0.40479, 'valueY': 0.15647, 'valueZ': 0.53959},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.108, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.28871, 'valueE': 0.20529, 'valueF': 0.09503, 'valueG': 0.08745, 'valueH': 0.07908, 'valueI': 0.14283},
    ],
    weights=[
        0.22581,
        0.13384,
        0.629,
        0.01135,
    ],
    issues=issues,
    reserved_value=0.4,
)