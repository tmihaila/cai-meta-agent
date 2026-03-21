from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain30

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueE", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.07974, 'valueB': 0.34596, 'valueC': 0.39291, 'valueD': 0.62285, 'valueE': 0.16051, 'valueF': 0.13257, 'valueG': 0.16992, 'valueH': 0.05287, 'valueI': 1.0, 'valueJ': 0.24879, 'valueK': 0.0, 'valueL': 0.01813, 'valueM': 0.06634, 'valueN': 0.12795},
        {'valueA': 0.21384, 'valueB': 0.00117, 'valueC': 0.29198, 'valueD': 0.58484, 'valueE': 0.07726, 'valueF': 0.02568, 'valueG': 0.32054, 'valueH': 0.0, 'valueI': 1.0, 'valueJ': 0.1552, 'valueK': 0.07931, 'valueL': 0.04057, 'valueM': 0.66252, 'valueN': 0.0702, 'valueO': 0.05981, 'valueP': 0.23646, 'valueQ': 0.27479, 'valueR': 0.02706, 'valueS': 0.11017, 'valueT': 0.27366, 'valueU': 0.13171, 'valueV': 0.14727, 'valueW': 0.17357},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.39997, 'valueC': 0.0},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.26527,
        0.13678,
        0.28728,
        0.05349,
        0.25718,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.49162, 'valueB': 0.0, 'valueC': 0.32012, 'valueD': 0.33276, 'valueE': 0.22875, 'valueF': 0.14005, 'valueG': 0.00193, 'valueH': 0.02876, 'valueI': 0.06324, 'valueJ': 0.46144, 'valueK': 0.10237, 'valueL': 0.30879, 'valueM': 1.0, 'valueN': 0.19575},
        {'valueA': 0.39945, 'valueB': 0.15469, 'valueC': 0.41363, 'valueD': 0.0828, 'valueE': 0.02376, 'valueF': 0.0, 'valueG': 0.14164, 'valueH': 0.26226, 'valueI': 0.19363, 'valueJ': 0.29022, 'valueK': 0.387, 'valueL': 0.57052, 'valueM': 0.8017, 'valueN': 0.07328, 'valueO': 0.05185, 'valueP': 0.00359, 'valueQ': 0.00139, 'valueR': 0.12787, 'valueS': 0.03168, 'valueT': 0.47513, 'valueU': 1.0, 'valueV': 0.26998, 'valueW': 0.37282},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.73152, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.26679,
        0.27828,
        0.02096,
        0.24134,
        0.19263,
    ],
    issues=issues,
    reserved_value=0.4,
)