from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain32

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.05858, 'valueB': 0.11854, 'valueC': 0.02953, 'valueD': 0.2683, 'valueE': 0.01513, 'valueF': 0.00838, 'valueG': 0.0, 'valueH': 0.01975, 'valueI': 1.0, 'valueJ': 0.00188, 'valueK': 0.28882, 'valueL': 0.31248, 'valueM': 0.15989, 'valueN': 0.07536, 'valueO': 0.06897, 'valueP': 0.09393, 'valueQ': 0.04013, 'valueR': 0.03639, 'valueS': 0.33793, 'valueT': 0.28403, 'valueU': 0.17975, 'valueV': 0.01677},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 0.29485, 'valueC': 1.0, 'valueD': 0.31298, 'valueE': 0.57319},
        {'valueA': 0.10177, 'valueB': 0.23263, 'valueC': 0.15325, 'valueD': 0.84267, 'valueE': 0.28859, 'valueF': 0.03289, 'valueG': 0.17918, 'valueH': 1.0, 'valueI': 0.09875, 'valueJ': 0.44311, 'valueK': 0.0, 'valueL': 0.5534, 'valueM': 0.16294},
    ],
    weights=[
        0.10116,
        0.20054,
        0.40681,
        0.29149,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.38517, 'valueB': 0.37922, 'valueC': 0.18108, 'valueD': 0.30803, 'valueE': 0.20069, 'valueF': 0.21953, 'valueG': 0.06756, 'valueH': 0.16787, 'valueI': 0.41722, 'valueJ': 0.23351, 'valueK': 0.89328, 'valueL': 0.44494, 'valueM': 0.17289, 'valueN': 0.95088, 'valueO': 0.05111, 'valueP': 1.0, 'valueQ': 0.35899, 'valueR': 0.18509, 'valueS': 0.56471, 'valueT': 0.41158, 'valueU': 0.0, 'valueV': 0.05305},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.79842, 'valueD': 0.38261, 'valueE': 0.40953},
        {'valueA': 0.39193, 'valueB': 0.83372, 'valueC': 0.20695, 'valueD': 0.37392, 'valueE': 0.18438, 'valueF': 0.23294, 'valueG': 0.10521, 'valueH': 1.0, 'valueI': 0.56356, 'valueJ': 0.21642, 'valueK': 0.33984, 'valueL': 0.28111, 'valueM': 0.0},
    ],
    weights=[
        0.138,
        0.50226,
        0.18786,
        0.17188,
    ],
    issues=issues,
    reserved_value=0.4,
)