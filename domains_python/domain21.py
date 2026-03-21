from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain21

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.27001, 'valueB': 1.0, 'valueC': 0.60474, 'valueD': 0.2426, 'valueE': 0.62822, 'valueF': 0.0},
        {'valueA': 0.37563, 'valueB': 0.05762, 'valueC': 0.0, 'valueD': 0.57174, 'valueE': 0.33491, 'valueF': 0.08902, 'valueG': 0.02323, 'valueH': 0.06842, 'valueI': 0.07137, 'valueJ': 1.0, 'valueK': 0.06863, 'valueL': 0.11767, 'valueM': 0.00468, 'valueN': 0.06979, 'valueO': 0.02449, 'valueP': 0.56284, 'valueQ': 0.18383, 'valueR': 0.14849, 'valueS': 0.06278, 'valueT': 0.23235, 'valueU': 0.05156, 'valueV': 0.12347, 'valueW': 0.08781, 'valueX': 0.11014, 'valueY': 0.3712, 'valueZ': 0.0346},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.32137},
        {'valueA': 0.07911, 'valueB': 0.01614, 'valueC': 0.46492, 'valueD': 0.19821, 'valueE': 1.0, 'valueF': 0.57441, 'valueG': 0.0},
    ],
    weights=[
        0.0178,
        0.13633,
        0.60106,
        0.24481,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.11222, 'valueC': 0.21438, 'valueD': 0.27849, 'valueE': 0.0, 'valueF': 0.13178},
        {'valueA': 0.26517, 'valueB': 0.11744, 'valueC': 0.32831, 'valueD': 0.05187, 'valueE': 0.2034, 'valueF': 0.13591, 'valueG': 0.64076, 'valueH': 0.54777, 'valueI': 0.02786, 'valueJ': 0.40464, 'valueK': 0.20546, 'valueL': 0.16092, 'valueM': 0.25311, 'valueN': 0.0, 'valueO': 0.20732, 'valueP': 0.00603, 'valueQ': 0.31052, 'valueR': 1.0, 'valueS': 0.26244, 'valueT': 0.10245, 'valueU': 0.23488, 'valueV': 0.00292, 'valueW': 0.09324, 'valueX': 0.16658, 'valueY': 0.18368, 'valueZ': 0.26468},
        {'valueA': 0.0, 'valueB': 0.46288, 'valueC': 1.0},
        {'valueA': 0.0611, 'valueB': 0.13232, 'valueC': 0.82467, 'valueD': 0.12919, 'valueE': 0.0, 'valueF': 1.0, 'valueG': 0.28658},
    ],
    weights=[
        0.21529,
        0.03263,
        0.50809,
        0.24399,
    ],
    issues=issues,
    reserved_value=0.4,
)