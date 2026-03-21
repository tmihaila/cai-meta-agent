from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain31

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.77602, 'valueB': 0.20285, 'valueC': 0.0, 'valueD': 0.48891, 'valueE': 0.34229, 'valueF': 0.09854, 'valueG': 0.05377, 'valueH': 0.16766, 'valueI': 0.01252, 'valueJ': 0.03343, 'valueK': 0.9472, 'valueL': 1.0, 'valueM': 0.05581, 'valueN': 0.07658, 'valueO': 0.50017, 'valueP': 0.06503, 'valueQ': 0.35383, 'valueR': 0.10537, 'valueS': 0.29274, 'valueT': 0.43246, 'valueU': 0.07341, 'valueV': 0.03223, 'valueW': 0.18561, 'valueX': 0.48806},
        {'valueA': 0.47782, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.69333},
        {'valueA': 0.0, 'valueB': 0.35457, 'valueC': 0.09421, 'valueD': 0.9434, 'valueE': 0.49646, 'valueF': 0.1479, 'valueG': 0.79973, 'valueH': 0.295, 'valueI': 1.0, 'valueJ': 0.15563, 'valueK': 0.22102, 'valueL': 0.22314, 'valueM': 0.26142, 'valueN': 0.39266, 'valueO': 0.73104, 'valueP': 0.20105, 'valueQ': 0.032},
    ],
    weights=[
        0.1781,
        0.51013,
        0.3053,
        0.00647,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.67091, 'valueB': 0.19761, 'valueC': 0.32215, 'valueD': 0.43334, 'valueE': 0.02344, 'valueF': 0.03322, 'valueG': 0.0, 'valueH': 0.81695, 'valueI': 1.0, 'valueJ': 0.86526, 'valueK': 0.33836, 'valueL': 0.06237, 'valueM': 0.76161, 'valueN': 0.09763, 'valueO': 0.527, 'valueP': 0.846, 'valueQ': 0.1279, 'valueR': 0.09192, 'valueS': 0.92295, 'valueT': 0.14451, 'valueU': 0.64849, 'valueV': 0.15633, 'valueW': 0.36812, 'valueX': 0.65114},
        {'valueA': 0.15791, 'valueB': 1.0, 'valueC': 0.13599, 'valueD': 0.0},
        {'valueA': 0.11259, 'valueB': 0.39871, 'valueC': 0.03691, 'valueD': 0.03551, 'valueE': 0.00669, 'valueF': 0.14194, 'valueG': 0.0, 'valueH': 1.0, 'valueI': 0.09723, 'valueJ': 0.11202, 'valueK': 0.07363, 'valueL': 0.60953, 'valueM': 0.24334, 'valueN': 0.07779, 'valueO': 0.08854, 'valueP': 0.39205, 'valueQ': 0.0332},
    ],
    weights=[
        0.27486,
        0.056,
        0.11842,
        0.55072,
    ],
    issues=issues,
    reserved_value=0.4,
)